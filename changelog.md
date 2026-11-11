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
| `0.3.0` | Phase 3 | Model loading, config inference, text and adapters | Completed (2026-11-08) |
| `0.4.0` | Phase 4 | Probe: hooks, builders, reduction, generation, recorder | Completed (2026-11-08) |
| `0.5.0` | Phase 5 | Renderer foundation, navigation, tiles and generic scene | Completed (2026-11-08) |
| `0.6.0` | Phase 6 | Dedicated scenes: matmul, attention, residual and prediction | Completed (2026-11-08) |
| `0.7.0` | Phase 7 | Presentation UX, widgets, nerd mode and polish | Completed (2026-11-08) |
| `1.0.0` | Phase 8 | Windows lecture-hall readiness, preflight, packaging and docs | Completed (2026-11-08) |
| `1.1.0` | Phase 9 | Crisp High-DPI rendering, auto-resolution & 1080p viewport | Completed (2026-11-11) |
| `1.2.0` | Phase 10 | Interactive trace launcher GUI and portable USB executables | Completed (2026-11-11) |

---

## [1.2.0] - 2026-11-11 — Phase 10: Interactive trace launcher GUI and portable USB executables

**Requirements addressed:** R1, R2, R5, R8, R9, R10, R11

### Added
- `src/tilevis/render/launcher.py`:
  - `TraceEntry` dataclass encapsulating path, filename, category (`User Trace` vs `Sample`), and file size in KB.
  - `discover_traces(traces_dir=None) -> list[TraceEntry]`: dynamically scans traces with priority ordering (explicit directory > frozen executable-adjacent `traces/` > project root `traces/` > `_MEIPASS` / package dir). Partitions entries into User Traces and Sample Traces, sorting user traces first, followed by samples, alphabetically.
  - `TraceMetadata` dataclass and fast `read_trace_metadata(path)` reader: extracts `__manifest__.npy` without unpacking heavy tensor arrays (< 20 ms), reading model name, layer count, embedding dimension, attention heads, parameter count, pass count, total steps, and prompt preview.
  - `LauncherApp`: full-featured `pygame-ce` GUI with scrollable trace dropdown (supporting mouse wheel, up/down arrows, and auto-scrolling to keep active item in view), metadata preview card, and interactive pills to toggle Window Mode (`borderless` / `windowed`), Theme (`dark` / `light`), Detail Level (`Story` / `Standard` / `Deep`), and Language (`EN` / `KO` / `JA`) before launching.
  - Trilingual UI localization (`_tr()`) in English, Korean, and Japanese using bundled Noto Sans fonts.
- `src/tilevis/cli.py`:
  - Added `launch` subcommand (`tilevis launch [--traces-dir PATH] [--profile NAME] [--headless]`).
  - Configured zero-argument invocation (`args.subcommand is None`) to automatically route to `_launch_command(args)`, opening the launcher GUI by default.
- Packaging & Standalone Executables:
  - `packaging/tilevis_player.spec`: upgraded PyInstaller spec supporting both Windows (`tilevis_player.exe`) and macOS (`BUNDLE` `tilevis_player.app` + Unix CLI binary). Configured with `hiddenimports` including `tilevis.render.launcher`.
  - `scripts/build_usb_dist.py`: automated script that builds the native binary using PyInstaller, sets isolated `PYINSTALLER_CONFIG_DIR`, and stages a ready-to-copy USB thumb-drive directory (`dist/usb_drive/tilevis_player/`) with the standalone binary/bundle, external editable `traces/` folder, presenter settings, and `README.txt`.
  - `scripts/build_macos.sh` & `scripts/build_windows.bat`: one-click native platform build scripts.
  - `.github/workflows/build_executables.yml`: multi-platform GitHub Actions CI workflow building native Windows and macOS standalone executables and uploading downloadable release artifacts.
- Tests:
  - `tests/render/test_launcher.py`: 11 unit tests and weakness probes covering trace discovery and priority sorting, fast metadata extraction, corrupt file safety, option toggles, keyboard navigation, mouse hit-testing, CLI routing, empty directory handling, and dropdown scrolling overflow.

### Changed
- `pyproject.toml` and `src/tilevis/__init__.py`: version bumped to `1.2.0`.
- `tests/test_cli.py`: updated `test_cli_help` to check for `launch` subcommand and updated zero-argument CLI test to verify routing to the launcher GUI.
- `packaging/README.md`, `README.md`, and `technical_document.md`: expanded documentation to cover the new Launcher GUI, portable USB thumb-drive layout, and cross-platform build instructions.

### Verification Summary
- **Tests passing:** 408 tests total (397 existing regression tests + 11 new launcher tests).
- **Test coverage:** 90% overall (core 88–91%, narration 90%, render 89%).
- **Boundary enforcement:** `tests/test_import_boundaries.py` and `tests/render/test_torch_free_suite.py` verified 100% compliant; zero PyTorch imports in render/launcher.
- **Static checks:** `ruff check` and `ruff format --check` passed across all 150 files; `mypy --strict` passed on `core`, `narration`, and `render`.
- **Immutability:** `git diff ch07/` verified completely empty.
- **Executable verified:** `dist/usb_drive/tilevis_player/` built and validated on macOS; standalone binary ran `check` and `launch --headless` with return code 0.

---

## [1.1.0] - 2026-11-11 — Phase 9: Native High-DPI rendering, automatic resolution detection, and crisp 1080p viewport

**Requirements addressed:** R1, R5, R9, R10, R11

### Added
- `src/tilevis/render/platform.py`:
  - `DisplayInfo` dataclass exposing `.desktop_size`, `.default_windowed`, `.desktop_w`, and `.desktop_h`.
  - `detect_display_info(display: int = 0) -> DisplayInfo` detecting desktop geometry and computing optimal windowed geometry.
  - `default_windowed_size(desktop_w: int, desktop_h: int) -> tuple[int, int]` generating crisp 16:9 resolutions (~85% of desktop bounds, minimum 800x450; exact 1920x1080 for monitors exceeding 1080p in both dimensions), replacing the fractional (1600, 900) default.
- `src/tilevis/render/window.py`:
  - Native High-DPI `pygame.Window` integration (`allow_high_dpi=True`) using `window.get_surface()` and `window.flip()`, with seamless fallback to `pygame.display.set_mode()` for legacy video environments.
  - Tracking of `logical_size`, `drawable_size`, and `dpi_scale` (`(drawable_w / logical_w, drawable_h / logical_h)`).
  - `WindowManager.map_point(pos) -> tuple[int, int]` accurately translating mouse positions from logical window space to physical backing pixels.
  - `WindowManager.present()` and `WindowManager.destroy()` lifecycle methods.
- `src/tilevis/render/animation.py`: `Timeline.seek(progress: float)` allowing preservation of fractional animation progress across viewport resizes.
- `tests/render/test_phase9_resolutions_and_cjk.py`: Parametrized test suite verifying compatibility across all 8 `COMPAT_MATRIX` architectures and all 6 committed sample traces (`big_en`, `big_es`, `big_ja`, `big_ko`, `tiny_ch07`, `tiny_demo`) across standard 1080p, Retina (`1512x982` / `3024x1964`), 1440p, and 4K viewports, plus `ch07/` immutability assertion.

### Changed
- `src/tilevis/render/platform.py`: `enable_high_dpi()` updated to set `SDL_WINDOWS_DPI_AWARENESS=permonitorv2` and `SDL_VIDEO_HIGHDPI_DISABLED=0` SDL hints before display init, and to invoke `SetProcessDpiAwarenessContext(-4)` (`DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2`) before falling back to `shcore.SetProcessDpiAwareness(2)` and `user32.SetProcessDPIAware()`.
- `src/tilevis/render/layout.py`: In `UiScale.from_viewport()`, snapped `s` to exact `1.0` when `abs(s - 1.0) < 0.01` (e.g. 1920x1080 or 1920x1200 viewports) so `px()` and `font_px()` yield exact unscaled integer pixels with zero rounding drift.
- `src/tilevis/render/app.py`:
  - `enable_high_dpi()` called at the beginning of `App.__init__()` prior to `pygame.font.init()`.
  - Default windowed size dynamically resolved via `detect_display_info()` when `viewport_size` is omitted.
  - Mouse motion and click events mapped via `self.window.map_point(raw_pos)`.
  - Live resize and `F11` fullscreen toggling immediately refresh `SceneContext` and re-render `TileGrid`s and fonts at the new physical pixel resolution while preserving fractional animation progress.
  - Frame presentation routed through `self.window.present()`, and window destruction handled via `self.window.destroy()`.
- `src/tilevis/render/scenes/common.py`: Removed redundant `pygame.draw.aalines()` pass over thick polylines in `draw_bezier_arc()`, eliminating blurred center seams down attention arcs.
- `src/tilevis/check.py`: Updated `check_display_modes()` to report detected display size, default windowed size, and High-DPI backing support via `pygame.Window`.
- `pyproject.toml` & `src/tilevis/__init__.py`: Version bumped from `1.0.0` to `1.1.0`.

### Verification & Test Summary
- **Automated tests:** **387 passed** in **56.52 s** (`pytest -m "not manual" --cov=tilevis`), **90%** overall statement coverage across 8,188 statements (`core` 88-100%, `narration` 90%, `render` 81-100%).
- **PyTorch-free playback boundary:** Verified with `test_torch_free_suite.py` and `test_import_boundaries.py` that all playback modules (`core`, `narration`, `render`, `check`) execute with `torch` completely absent from `sys.modules`.
- **Static checks:** `ruff check src tests` clean, `ruff format --check src tests` clean (147 files), `mypy --strict` on `core`/`narration`/`render` (54 files clean), `mypy` on `probe` and `check.py` (33 files clean), and `git diff ch07/` completely empty.
- **Resolution & HiDPI probes:** Simulated 2.0x Retina (`1512x982` window -> `3024x1964` drawable) and 1.5x Windows HiDPI (`1280x720` window -> `1920x1080` drawable) click coordinate mapping validated against interactive buttons; live resize across 720p, 1080p, and Retina verified without stale surfaces or lost state.

---

## [Unreleased]

### Added
- `src/tilevis/narration/locales/ko.toml`: Complete Korean narration catalog translating all 19 `Semantic` keys, 14 `StepKind` fallback keys, and synthetic title/end cards into intuitive, presentation-friendly Korean text while preserving exact `{placeholders}` for model metrics.
- `src/tilevis/render/text.py`: Automatic Korean font fallback in `FontManager` detecting Hangul characters (`\uac00-\ud7af`, `\u1100-\u11ff`, `\u3130-\u318f`) and dynamically resolving platform Korean fonts (`AppleSDGothicNeo.ttc` / `AppleGothic.ttf` on macOS, `malgun.ttf` / `malgunbd.ttf` on Windows, `NotoSansCJK-Regular.ttc` / `NanumGothic.ttf` on Linux) or bundled `NotoSansKR-*.ttf` assets, with safe fallback to `NotoSans`.
- `src/tilevis/render/commands.py`: `Command.TOGGLE_LOCALE = "toggle_locale"` presenter command.
- `src/tilevis/render/input_map.py`: Configurable `L` key binding mapped to `Command.TOGGLE_LOCALE`.
- `src/tilevis/render/widgets/buttons.py`: Interactive on-screen `Lang: EN` / `Lang: KO` toggle button with active state highlighting in Korean mode.
- `src/tilevis/render/widgets/help_overlay.py`: Added `Language Toggle (EN / KO)` entry (`L`) to the presenter shortcut cheat sheet.
- Test suites: `tests/render/test_korean_locale_and_toggle.py` (6 unit and headless integration tests covering translations, font manager rendering, keybinding, button interaction, and full playback sweeps), plus parametrized coverage in `tests/narration/test_captions.py`.

### Changed
- `src/tilevis/render/app.py`: Implemented `_toggle_locale()` method switching active `CaptionProvider` between `"en"` and `"ko"`, clearing font caches, and updating scene captions mid-presentation.
- `src/tilevis/render/widgets/hud.py`: Expanded footer buttons area to accommodate 6 control buttons comfortably without clipping.
- `presenter/default.toml` & `README.md`: Documented `toggle_locale = ["l"]` keybinding and on-screen button controls.

---

**Requirements addressed:** R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11

### Added
- `src/tilevis/check.py`: Comprehensive preflight environment and trace diagnostic suite implementing `check_python_version()`, `check_os_and_dpi()`, `check_pygame_packages()`, `check_bundled_fonts()`, `check_display_modes()`, `check_trace_file()`, and `run_checks()`. Features collision detection against legacy `pygame`, Windows High-DPI per-monitor awareness detection, bundled font file integrity verification, desktop display query, and trace telemetry (model spec, passes, steps, load time, and fidelity status).
- `tilevis check`: Fully wired CLI subcommand reporting preflight status with actionable diagnostics, returning exit code 0 on all passes or 1 on any failure.
- `scripts/play_demo.bat`: Lecture-hall Windows batch launcher configured with `chcp 65001` (UTF-8 console), preflight environment validation, demo playback launch, and error pause.
- `scripts/play_demo.sh`: Cross-platform Unix/macOS shell script launcher for demo playback.
- `packaging/tilevis_player.spec`: PyInstaller single-folder specification bundling only playback modules (`core`, `narration`, `render`, `check`), bundled fonts, sample traces, and default presenter configs, while explicitly excluding heavy recording packages (`torch`, `tiktoken`, `matplotlib`, `pytest`, `hypothesis`, `mypy`).
- `packaging/README.md`: Contributor guide for generating standalone Windows executables with PyInstaller.
- `README.md`: Complete presentation quick start guide including 60-second quick start, teammate profile & record guide, presenter cheat sheet (keyboard & mouse), and lecture-hall troubleshooting table.
- Test suites: `tests/test_phase8_check_and_packaging.py` (10 tests covering clean preflight checks, missing trace, corrupt trace, missing font asset simulation, package collision diagnostics, PyInstaller spec structure & exclusions, clean-room Korean/space path playback, read-only directory playback, focus loss and F11 toggling, and CLI user-error hygiene).

### Changed
- `pyproject.toml` & `src/tilevis/__init__.py`: bumped version from `0.7.0` to `1.0.0`.
- `tests/test_cli.py`: updated `test_cli_check_stub` to `test_cli_check_implemented` validating real preflight check CLI output and zero exit code.

### Fixed
- **Pytest output capture for preflight diagnostics:** Changed default `out: TextIO = sys.stdout` in `run_checks()` to dynamic `out is None: out = sys.stdout` lookup to ensure proper compatibility with test capture fixtures (`capsys`) and avoid binding to early import-time streams.
- **Clean-room playback robustness:** Validated seamless playback from directories with Korean characters (`"2026 발표 자료/발표용 traces"`), spaces, and read-only media (locked USB drive simulation) with zero write attempts.
- **Window focus & mode toggling hygiene:** Verified that window focus loss (`WINDOWFOCUSLOST`) clears pressed key modifiers without resetting presentation cursor position, and F11 fullscreen toggle preserves exact step state.

### Verification & Test Summary
- **Automated tests:** **318 passed** in **38.37 s** (`pytest -m "not manual" --cov=tilevis`, including `slow` and `perf` suites; **10** new tests added in `tests/test_phase8_check_and_packaging.py`), **91%** overall statement coverage across 7,916 statements (`core` 91%, `narration` 90%, `render` 91%, `check.py` 84%).
- **PyTorch-free playback boundary:** Verified with `test_torch_free_suite.py` and `test_import_boundaries.py` that all playback modules (`core`, `narration`, `render`, `check`) execute with `torch` completely absent from `sys.modules`.
- **Static checks:** `ruff check src tests` clean, `ruff format --check src tests` clean (145 files), `mypy --strict` on `core`/`narration`/`render` (54 files clean), `mypy` on `probe` and `check.py` (33 files clean), and `git diff ch07/` completely empty.

---

## [0.7.0] - 2026-11-08 — Phase 7: Presentation UX, Widgets, Nerd Mode and Polish

**Requirements addressed:** R1, R2, R4, R5, R7, R8, R9, R10, R11

### Added
- `src/tilevis/render/widgets/minimap.py`: `MinimapWidget` rendering bounded pipeline overview bar (`Input ▸ B1 ▸ B2 … ▸ Output`), dynamic middle-block collapsing with ellipses for deep architectures (`12`, `24`, `48` layers), active stage/block highlight, and direct `JUMP` hit testing.
- `src/tilevis/render/widgets/buttons.py`: `ButtonsWidget` on-screen clickable navigation controls (`< Prev`, `Next >`, `Detail` level pill, `Nerd` mode toggle, `?` help button) with hover states enabling mouse-only presentation.
- `src/tilevis/render/widgets/legend.py`: `LegendWidget` colormap gradient bar with formatted numeric endpoints (`-scale / 0 / +scale` for diverging maps, `0% / 50% / 100%` for sequential maps).
- `src/tilevis/render/widgets/tooltip.py`: `TooltipWidget` floating hover card displaying original unreduced coordinates (token text, feature/head index, candidate word) and exact float values, with four-corner edge flip and clamping guaranteeing coordinates stay within `[0, W] × [0, H]`.
- `src/tilevis/render/widgets/nerd_panel.py`: `NerdPanelWidget` toggled via `N` displaying technical telemetry (step ID, module path, semantic, tensor shapes, reduction strategy, full tensor statistics, Fidelity status badge with max error, normalizer scale, and close button).
- `src/tilevis/render/widgets/help_overlay.py`: `HelpOverlayWidget` modal keyboard & mouse cheat sheet toggled with `H`/`F1` or on-screen `?` button, generated dynamically from active `InputMap` keybindings, with backdrop dismiss and modal precedence.
- `tilevis demo`: In-memory seeded demo subcommand (`tilevis demo [--prompt TEXT] [--theme THEME] [--detail DETAIL] [--headless]`) building `tiny_random`, recording 3 passes, and launching interactive playback.
- Test suites: `tests/render/test_widgets.py` (13 tests for minimap, buttons, tooltip four-corner clamping, nerd panel fidelity styling, legend endpoints, WCAG contrast), `tests/render/test_phase7_ux_and_weakness.py` (12 tests for overlay precedence, mouse-only presentation walkthrough, 50-toggle theme cache invalidation, and compat matrix walkthrough), and `tests/integration/test_demo_cli.py` (2 integration tests for `tilevis demo`).

### Changed
- `pyproject.toml` & `src/tilevis/__init__.py`: bumped version from `0.6.0` to `0.7.0`.
- `src/tilevis/render/widgets/hud.py`: integrated `MinimapWidget` into header center and `LegendWidget` + `ButtonsWidget` into footer.
- `src/tilevis/render/app.py`: wired widget event routing, mouse movement / hover tracking, click consumption, modal overlay precedence, live theme cache invalidation, and autoplay (`P`).
- `src/tilevis/cli.py`: implemented `tilevis demo` subcommand with `--headless`, `--theme`, and `--detail` flags.
- `tests/test_cli.py`: updated `demo` subcommand test from stub expectation to implemented execution.

### Fixed
- **Mouse event routing & stage isolation:** Ensured clicking on-screen buttons (`< Prev`, `Next >`, `Detail`, `Nerd`, `?`) and minimap segments consumes the mouse event and triggers only that widget's action without triggering accidental stage navigation.
- **Modal overlay precedence:** Guaranteed that while `HelpOverlayWidget` is open, any click or `Escape` keypress dismisses the overlay without advancing steps or arming application exit.
- **Four-corner tooltip viewport bounding:** Verified that tooltips triggered at exact screen corners `(0, 0)`, `(W-1, 0)`, `(0, H-1)`, and `(W-1, H-1)` flip and clamp strictly inside `[0, W] × [0, H]` across all viewport sizes (`800x600`, `1920x1080`, `3440x1440`).
- **Theme font and surface invalidation:** Re-created `FontManager` and invalidated scene tile surfaces on `T` toggle to ensure clean color switches with zero memory leaks.

### Decisions & Rationale
- **Self-contained footer controls for lecture-hall presentation:** Placed `< Prev`, `Next >`, detail pill, nerd badge, and help button directly on the footer alongside the color legend, enabling seamless mouse-only control on a lecture-hall PC without requiring a physical keyboard.
- **Dynamic minimap windowing for deep models:** Kept minimap segment count bounded ($\le 14$ segments) for 12, 24, and 48 layer models by pinning input, output, first/last blocks, and a sliding 3-block window around the active block with ellipses.

### Verification & Test Summary
- **Automated tests:** **308 passed** in **35.16 s** (`pytest -m "not manual" --cov=tilevis`, including `slow` and `perf` suites; **27** new tests added across `tests/render/` and `tests/integration/`), **91%** overall statement coverage across 7,761 statements (`tooltip.py` 100%, `help_overlay.py` 99%, `nerd_panel.py` 99%, `legend.py` 98%, `hud.py` 94%, `buttons.py` 92%, `app.py` 80%, `minimap.py` 82%).
- **Compatibility matrix & torch-free guarantee:** Walked through all 8 `compat_matrix` models clicking buttons and toggling `N`, `T`, `D`, `H`, and `P`; verified `tilevis demo` smoke tests; verified `test_torch_free_suite.py` and `test_import_boundaries.py` pass cleanly.
- **WCAG contrast compliance:** Automated test verified relative-luminance contrast ratio $\ge 4.5:1$ for all primary and secondary text against background, panel, and panel_alt colors in both `dark` (7.38–16.97:1) and `light` (7.29–17.58:1) themes.
- **Static checks:** `ruff check src tests` clean, `ruff format --check src tests` clean (143 files), `mypy --strict` on `core`/`narration`/`render` (54 files clean), `mypy` on `probe` (32 files clean), `test_import_boundaries.py` clean, and `git status/diff ch07/` completely empty.
- **Manual checks:** Generated and visually inspected 6 UI widget captures (`01_default_hud_minimap_buttons_legend.png`, `02_nerd_mode_panel.png`, `03_tooltip_hover.png`, `04_help_overlay.png`, `05_light_theme_hud.png`, and `06_800x600_compact_hud.png`), confirming crisp text legibility, balanced minimap layout, properly aligned legend, and cleanly formatted help overlay.

### Known Issues & Deferred Items
- Standalone lecture-hall packaging, `tilevis check` preflight diagnostics, `scripts/play_demo.bat` / `scripts/play_demo.sh`, and `README.md` are scheduled for Phase 8 (`1.0.0`).

---

## [0.6.0] - 2026-11-08 — Phase 6: Dedicated Scenes: Matmul, Attention, Residual and Prediction

**Requirements addressed:** R1, R2, R3, R4, R5, R7, R8, R9, R10, R11

### Added
- `src/tilevis/render/scenes/common.py`: `BaseScene`, `GridCard`, `HitRegion`, cached `draw_grid_card()` with $O(1)$ rectangular sweep reveal, geometric operators (`draw_arrow`, `draw_operator_symbol`, `draw_bezier_arc`), `place_grid_in_card()`, and `format_token_chip()`.
- Dedicated animated `Scene` implementations registered via `@SCENES.register(...)` across all 14 `StepKind` values plus synthetic `TITLE` and `END` cards:
  - `title.py`: `TitleScene` (`TITLE`, prompt preview token pills and friendly layer/parameter badge summary) and `EndScene` (`END`, full generated sequence highlighted by pass with EOS indicator).
  - `tokenize.py`: `PromptScene` (`PROMPT`, framed prompt card) and `TokenizeScene` (`TOKENIZE`, source text splitting into colored token chips with ID badges).
  - `embedding.py`: `EmbeddingScene` (`EMBEDDING`, color-coded token shelf fanning into `token_vectors` + `position_vectors` -> `Word Meaning`, supporting `pos_emb=None` and dense `emb_dim` views).
  - `layernorm.py`: `LayerNormScene` (`LAYERNORM`, before/after activation cards with spread gauges and `scale`/`shift` parameter strips).
  - `matmul.py`: `MatmulScene` (`LINEAR`, `(T, in) × (in, out) = (T, out)` matrix multiplication with optional `bias` strip, synchronized column-sweep highlight across weights and outputs, and semantic framing for `att.out`, `ff.expand`, and `ff.contract`).
  - `qkv.py`: `QkvProjectionScene` (`QKV_PROJECTION`, shared input branching through three weight panels into `Q` Question, `K` Label, and `V` Content tile grids).
  - `attention.py`: `AttentionPatternScene` (`ATTENTION_PATTERN`, pre-baked causal upper-triangle greyed $T \times T$ heatmap, curved spotlight attention arcs for the newest token, interactive head wrap `]` / `[`, and head-average toggle `A`).
  - `attention_mix.py`: `AttentionMixScene` (`ATTENTION_MIX`, attention-weighted mixing of `V` rows into context vectors).
  - `activation.py`: `ActivationScene` (`ACTIVATION`, pre/post GELU panels highlighting muted negative activations alongside a procedural GELU curve badge).
  - `residual.py`: `ResidualScene` (`RESIDUAL_ADD`, `shortcut` highway stream + `update` stream sliding together into summed output tiles).
  - `block_summary.py`: `BlockSummaryScene` (`BLOCK_SUMMARY`, block input -> output alongside the `output - input` delta grid for collapsed blocks).
  - `logits.py` & `prediction.py`: `LogitsScene` (`LOGITS`, final token's vector × top-k candidate rows of `out_head`) and `PredictionScene` (`NEXT_TOKEN`, staggered horizontal probability bars sorted by probability, sampled/greedy winner badge animating into the running sentence, and theme-colored confetti).
- Test suites: `tests/render/test_scenes_contract.py` (33 parameterized scene contract tests over all 16 `SCENES` keys × `dark`/`light` themes), `tests/render/test_scenes_specific.py` (6 scene behavior and edge-shape tests), `tests/render/test_phase6_matrix_and_perf.py` (27 tests covering `StepPlayer` animation-skip regression, all 8 `compat_matrix` configs across `STORY`/`STANDARD`/`DEEP`, 200-command rapid input mashing at `800×600`/`1920×1080`/`3440×1440`, and `@pytest.mark.perf` `1920×1080` enter/frame budgets), and `tests/integration/test_phase6_recorded_scenes.py` (8 real `TraceRecorder` -> headless playback integration tests including `@pytest.mark.slow` `gpt2_small_shape`).

### Changed
- `pyproject.toml` & `src/tilevis/__init__.py`: bumped version from `0.5.0` to `0.6.0`.
- `src/tilevis/render/scenes/registry.py`: loaded all 13 dedicated scene modules after `SCENES` creation so every `StepKind` self-registers cleanly while retaining `GenericScene` fallback.
- `src/tilevis/narration/locales/en.toml`: refined metaphor-driven captions and nerd-mode summaries to align with the visual scenes while preserving format placeholders.
- `tests/render/test_app_headless.py`: adjusted animation-completion tick budget from `1.0 s` to `2.0 s` to match Phase 6 scene durations (`0.85–1.35 s × speed`).

### Fixed
- **Dense-model `EMBEDDING` token shelf (`embedding.py`):** Replaced a `pitch >= 18` visibility cutoff that hid token chips at `emb_dim >= 48` with a stacked token shelf and color-coded connection strands (up to 12 tokens) so `small_bias` and `gpt2_small_shape` retain clear token-to-row attribution.
- **`ATTENTION_PATTERN` label & arc separation (`attention.py`):** Positioned spotlight percentage labels to the right of token pills and guarded bottom column headers (`cell_px >= 42`) to eliminate text-on-arc and adjacent-token overlap.
- **`LOGITS` `800×600` header clearance (`logits.py`):** Increased score-board header height (`sb_hdr_h`) so the logit tile strip stays cleanly separated from `"Tile scores per candidate"` on compact displays.
- **Small-probability readout clipping (`prediction.py`):** Formatted probabilities `< 0.005` (`0.5%`) as `.2%` and widened `pct_w` so low un-finetuned probabilities (e.g., `"0.02% [PICKED]"`) render legibly without right-edge clipping.
- **Tofu-free geometric operators (`common.py`):** Rendered arrows, `+`, `×`, `=`, and attention arcs using procedural `pygame.draw` primitives and ASCII/escaped labels so bundled `NotoSans` fonts avoid missing-glyph boxes.

### Decisions & Rationale
- **Pre-baked `TileGrid` cards + $O(1)$ reveal sweeps (`scenes/common.py`):** Static colormapped surfaces and causal masks are built once in `enter()`, while `update()`/`draw()` only update progress floats and rectangular clip sweeps. This keeps `1920×1080` worst-case `enter()` at `10.14 ms` (`< 100 ms` budget) and worst-case animation frames at `1.02 ms` (`< 16 ms` budget).
- **Additive self-registration via `scenes/registry.py`:** Each dedicated scene registers via `@SCENES.register(...)` on module import, avoiding `if/elif` branching while preserving seamless `GenericScene` fallback for unknown future `StepKind`s.

### Verification & Test Summary
- **Automated tests:** **281 passed** in **31.16 s** (`pytest -m "not manual" --cov=tilevis`, including `slow` and `perf` tests; **74** new tests added across `tests/render/` and `tests/integration/`), **91%** overall statement coverage across 7,134 statements (`prediction.py` 98%, `layernorm.py` 98%, `residual.py` 97%, `attention_mix.py` 97%, `block_summary.py` 97%, `title.py` 97%, `embedding.py` 96%, `qkv.py` 96%, `activation.py` 95%, `common.py` 95%, `attention.py` 94%, `matmul.py` 94%, `tokenize.py` 92%, `logits.py` 90%).
- **Compatibility matrix & torch-free guarantee:** Verified both synthetic traces (`STORY`/`STANDARD`/`DEEP` × 3 animation frames/step) and live `TraceRecorder` -> headless `App` playback across all 8 `compat_matrix` models (`tiny_ch07`, `small_bias`, `heads_8_of_64`, `head_dim_1`, `deep_narrow`, `lora_small`, `wrapped_ckpt`, and `@pytest.mark.slow` `gpt2_small_shape`); `tests/render/test_torch_free_suite.py` passed with `torch` blocked in `sys.meta_path`.
- **Performance (§8.4 budgets at `1920×1080` headless):** Verified across all 15 active playlist `StepKind`s on `gpt2_small`-scale reduced snapshots (`T=24`, `emb_dim=64`); worst initial `enter()` + layout = **10.14 ms** (`ACTIVATION`, `< 100 ms` budget); worst 20-frame animation average = **1.02 ms/frame** (`QKV_PROJECTION`, `< 16.0 ms` budget).
- **Static checks:** `ruff check src tests` clean, `ruff format --check src tests` clean (140 files formatted), `mypy --strict` on `core`/`narration`/`render` (54 files clean), `mypy` on `probe` (32 files clean), `tests/test_import_boundaries.py` clean, and `git status/diff ch07/` completely empty.
- **Manual checks:** Headlessly rendered and visually inspected **40 PNG frames** covering all 16 scene types (`t=mid` and `t=end` on `traces/samples/tiny_demo.trace.npz` across `dark` and `light` themes at `1920×1080` and `800×600`), catching and fixing the 4 layout/readout bugs recorded under *Fixed*. Live desktop window & screenshare rehearsal remains deferred to Phase 7/8 manual verification.

### Known Issues & Deferred Items
- Interactive HUD widgets (`minimap`, `legend`, `buttons`, hover `tooltip`, `nerd_panel`, `help_overlay`), live hotkeys (`T`, `+`/`-`, `P`), and `tilevis demo` are scheduled for Phase 7 (`0.7.0`).

---

## [0.5.0] - 2026-11-08 — Phase 5: Renderer Foundation, Navigation, Tiles and Generic Scene

**Requirements addressed:** R1, R2, R3, R4, R5, R7, R8, R9, R10, R11

### Added
- `src/tilevis/narration/`:
  - `locales/en.toml`: Audience-friendly title, metaphor-driven body captions, and technical nerd-mode templates for all 19 `Semantic` entries, all 14 `StepKind` fallbacks, plus synthetic `title` and `end` cards.
  - `captions.py`: `Caption` dataclass and `CaptionProvider` with `semantic -> kind -> generic` lookup hierarchy, dynamic `{ff_dim}` extraction, and non-crashing `?` substitution for missing placeholders.
- `src/tilevis/assets/fonts/`: Bundled SIL OFL 1.1 `NotoSans-Regular.ttf`, `NotoSans-Bold.ttf`, and `OFL.txt` with zero system font dependency.
- `src/tilevis/render/`:
  - `platform.py` & `window.py`: Windows High-DPI awareness (`SetProcessDpiAwareness(2)` with `SetProcessDPIAware()` fallback), `WindowManager` supporting `borderless`, `windowed`, and `exclusive` modes, `F11` mode toggling, programmatic 3×3 tile icon, and resize handling.
  - `themes/base.py` & `themes/builtin.py`: `Theme` dataclass and `THEMES` registry with colorblind-safe `dark` and `light` themes verified to $\ge 4.5:1$ WCAG contrast.
  - `colormap.py` & `normalization.py`: `COLORMAPS` (`diverging_blue_orange`, `sequential_purple`, `diverging_teal_magenta`) with bit-exact midpoint zero mapping at `lut[127]`, plus `NORMALIZERS` (`per_tensor`, `shared_step`, `global_pass`) using outlier-resistant `abs_p99`.
  - `text.py`: `FontManager` loading bundled OFL fonts via `importlib.resources` + `io.BytesIO` with size caching, pixel-accurate wrapping/truncation, and `.notdef` fallback substitution.
  - `layout.py`: `UiScale` logical `1920x1080` design space, aspect-preserving letterboxing, and non-overlapping `header`, `stage`, `caption`, and `footer` regions across extreme aspect ratios.
  - `tiles.py`: `TileGrid` vectorized NumPy RGB surface caching, automatic display downsampling (`"showing N of M"`) below minimum cell thresholds, exact `cell_at()` reverse coordinate mapping via `AxisInfo.indices`, and reveal/highlight overlays.
  - `playlist.py` & `player.py`: `StepRef`, `build_playlist()` (`STORY`/`STANDARD`/`DEEP`, `detail_blocks`, `later_pass_level`, synthetic `TITLE`/`END` cards), `resolve_step()`, and bounded `StepPlayer` with cursor re-anchoring on setting updates.
  - `commands.py` & `input_map.py`: `Command` StrEnum and `InputMap` supporting configurable TOML `[keys]` overrides, Caps/NumLock-tolerant modifiers, mouse click routing, and keycode resolution without requiring an initialized video system.
  - `animation.py`: `EASINGS` (`linear`, `ease_out_cubic`, `ease_in_out_quad`), `Tween`, and speed-scaled `Timeline` with immediate `finish()` completion.
  - `scenes/base.py`, `scenes/registry.py`, `scenes/generic.py`: `Scene` protocol, `SceneContext`, `Tooltip`, `SCENES` registry, and `GenericScene` fallback rendering 1-D/2-D/3-D tensors (with head cycling) and 0-tensor metadata cards.
  - `widgets/caption_box.py` & `widgets/hud.py`: Wrapped `CaptionBox` (including `[Nerd]` summary line) and `HudWidget` progress indicator, double-`Esc` arming warning, and keyboard hint footer.
  - `app.py`: `App` coordinating event dispatch, pure `tick(dt, events)`, 60 FPS cap, idle `pygame.event.wait(250)` throttling, and 2.0 s double-`Esc` quit confirmation.
- Test suites: `tests/narration/test_captions.py`, `tests/render/test_colormap_norm.py`, `tests/render/test_tiles.py`, `tests/render/test_playlist_player.py`, `tests/render/test_animation.py`, `tests/render/test_input_map.py`, `tests/render/test_themes_layout.py`, `tests/render/test_window_and_platform.py`, `tests/render/test_app_headless.py`, `tests/render/test_torch_free_suite.py`, and `tests/integration/test_play_cli.py`.

### Changed
- `pyproject.toml`: bumped version to `0.5.0` and configured `[tool.setuptools.package-data]` for `assets/fonts/*` and `narration/locales/*.toml`.
- `src/tilevis/__init__.py`: bumped `__version__` to `"0.5.0"`.
- `src/tilevis/cli.py`: implemented `tilevis play` subcommand (`--presenter`, `--window`, `--theme`, `--detail`, `--headless`) and updated `tests/test_cli.py`.
- `src/tilevis/core/validation.py`: reinforced `validate_trace()` to reject tensors with empty/zero-sized dimensions (`0 in shape`).
- `technical_document.md`: documented the non-empty tensor validation invariant and `--headless` CLI playback flag.

### Fixed
- **Empty candidate tensors in `probe/builders/output.py` (found during Phase 5 sample trace playback):** `OutputStepBuilder` previously passed raw vocabulary token IDs (`0..50256`) as `custom_indices_by_axis` into already-sliced `(emb_dim, k)` candidate weights/logits/probability arrays (`full_size = k`), stripping candidates with IDs `>= k` to zero-sized shapes `(48, 0)`, `(1, 0)`, `(0,)`. Removed erroneous `custom_indices_by_axis` so candidate dimensions retain `[0..k-1]` indices with matching word labels, stored `candidate_ids` in `lm_head` step metadata, added an empty-slice guard in `GenericScene`, reinforced `validate_trace()`, and regenerated `traces/samples/tiny_demo.trace.npz` and `traces/samples/tiny_ch07.trace.npz`.
- `App.run()` event polling when `headless=True`: avoided calling `pygame.event.get()`/`wait()` on an uninitialized SDL video subsystem so `tilevis play --headless` succeeds without requiring `SDL_VIDEODRIVER=dummy`.
- `Tween.value_at()` endpoint precision: clamped `elapsed >= delay + duration` directly to `end` against IEEE-754 subtraction rounding.
- HUD footer missing-glyph boxes in `NotoSans`: updated footer hint labels to clear text (`Right / Space: Next  ·  Left: Back ...`) and added cached `.notdef` fallback substitution in `FontManager` for arrow/math/newline symbols (`↵` -> `\n`).

### Decisions & Rationale
- **Bundled OFL font with safe symbol fallback:** Bundled `NotoSans-Regular.ttf` and `NotoSans-Bold.ttf` via `importlib.resources` (never OS system fonts) and added runtime `.notdef` pixel comparison caching so tokens containing `↵` or arrows render cleanly without tofu boxes while remaining compatible with richer custom fonts in the future.
- **Logical 1920×1080 scaling + automatic tile downsampling:** Scales layout proportionally across 800×600 through 4K and ultrawide 3440×1440 viewports, automatically switching tight `TileGrid` panels to `"showing N of M"` while preserving exact reverse index lookup in `cell_at()`.

### Verification & Test Summary
- **Automated tests:** **207 passed** in **17.04s** (`64` new tests added across `tests/narration/`, `tests/render/`, `tests/integration/test_play_cli.py`, and `tests/core/test_validation.py`), **90%** overall statement coverage across 4,590 statements (`src/tilevis/render/tiles.py` 95%, `generic.py` 95%, `captions.py` 90%).
- **Compatibility matrix & torch-free guarantee:** Headless `App` stepped start-to-end and back across synthetic traces for all 8 `compat_matrix` models (`tiny_ch07`, `small_bias`, `heads_8_of_64`, `head_dim_1`, `deep_narrow`, `lora_small`, `wrapped_ckpt`, `gpt2_small_shape`) and both regenerated Phase 4 sample traces; subprocess runner executed the entire `tests/render` + `tests/narration` suite with `torch` blocked in `sys.meta_path`.
- **Static checks:** `ruff check src tests` (clean), `ruff format --check src tests` (135 files formatted), `mypy --strict` on `core`, `narration`, `render` (53 files clean), `mypy` on `probe` (32 files clean), `test_import_boundaries.py` clean, and `git status/diff ch07/` completely empty.
- **Manual checks:** Verified real Cocoa SDL display creation on macOS (`pygame-ce 2.5.8 / SDL 2.32.10`) in both `windowed` (`1512x842`) and `borderless` (`1512x982`) modes, and visually inspected rendered `1920x1080` PNG captures (`title`, `embed`, `att.scores`, `next_token`) confirming crisp Noto Sans typography, legible tile grids, and clean captions. Interactive mouse/keyboard rehearsal on the target Windows machine will run alongside Phase 7/8 checks.

### Known Issues & Deferred Items
- Dedicated animated step scenes (`matmul`, `qkv`, `attention` heatmap + arcs, `residual`, `prediction` bar chart) are scheduled for Phase 6 (`0.6.0`); all step kinds render through `GenericScene` in `0.5.0`.
- Interactive UI widgets (`minimap`, `legend`, `buttons`, `tooltip`, `nerd_panel`, `help_overlay`) and `tilevis demo` are scheduled for Phase 7 (`0.7.0`).

---

## [0.4.0] - 2026-11-08 — Phase 4: Probe: Hooks, Builders, Reduction, Generation, Recorder

**Requirements addressed:** R1, R2, R3, R4, R6, R7, R8, R11

### Added
- `src/tilevis/probe/hooks.py`: `HookManager` context manager registering forward pre/post hooks on requested submodule paths, tracking per-module call counters, restoring model training state, and guaranteeing handle cleanup in `finally` even on exceptions.
- `src/tilevis/probe/capture.py`: `CaptureBuffer` storing unbatched, detached CPU `float32` tensors keyed by `(module_path, call_idx, role)` with batch dimension 0 removed.
- `src/tilevis/probe/reduce.py`: Dimensionality reduction engine, `REDUCERS` registry (`none`, `stride`, `block_mean`, `max_abs`), `compute_stride_indices()` generating strictly increasing in-bounds coordinate maps, `IndexPlan` maintaining cross-step alignment across axis families (`model_dim`, `ff_dim`, `head_dim`, `token`, `head`), `compute_tensor_stats()` computed on full unreduced tensors, and `create_tensor_snapshot()` with non-finite value sanitization.
- `src/tilevis/probe/builders/`: Modular step builders conforming to `StepBuilder` protocol and registered in `STEP_BUILDERS`:
  - `embedding.py`: `PROMPT`, `TOKENIZE`, and `EMBEDDING` step builders assembling token and position vector snapshots and their sum.
  - `layernorm.py`: `LAYERNORM` steps for transformer blocks and final normalization, with scale/shift weight extraction and mathematical fidelity grading.
  - `attention.py`: Attention builder performing multi-head scaled dot-product attention recomputation under `torch.inference_mode()`, causal masking, automatic `n_heads` divisor inference from `emb_dim`, numerical fidelity grading (`exact`, `approximate`, `unverified`, `failed`), emitting `QKV_PROJECTION`, `ATTENTION_PATTERN` ($H \times T \times T$), `ATTENTION_MIX`, and `LINEAR` (`block.att.out`), with graceful `GENERIC` fallback on failure.
  - `residual.py`: `RESIDUAL_ADD` steps with identity verification (`out ≈ shortcut + update`) for both attention and feedforward sublayers.
  - `feedforward.py`: MLP sublayer step construction for `LINEAR` (expand), `ACTIVATION` (GELU), and `LINEAR` (contract).
  - `block_summary.py`: `BLOCK_SUMMARY` step builder at `DetailLevel.STORY` computing `update = output - input`.
  - `output.py`: `LOGITS` step builder focusing on the last token's vector meeting top-k candidate weight rows, and `NEXT_TOKEN` step builder with candidate ranking from ground-truth forward pass logits.
  - `generic.py`: Dynamic fallback step builder mapping leaf modules for non-standard architectures under `GenericAdapter`.
- `src/tilevis/probe/generation.py`: Autoregressive generation loop supporting greedy argmax, top-k sampling with temperature and seeded reproducibility, context window left-cropping (`idx[:, -context_size:]`), and EOS termination.
- `src/tilevis/probe/recorder.py`: `TraceRecorder` orchestrating forward execution under hooks, executing step builders in deterministic template order, assembling `ModelSpecInfo`, `PassRecord`, and `Trace`, and strictly validating invariants with `validate_trace()`.
- `src/tilevis/cli.py`: Wired `tilevis record` subcommand with CLI options `--profile`, `--prompt`, `--new-tokens`, `--strategy`, `--temperature`, `--top-k`, `--seed`, `--n-heads`, `--device`, `--max-features`, `--max-tokens`, and `-o/--output`.
- `traces/samples/`: Committed compact reference sample traces:
  - `tiny_demo.trace.npz`: `small_bias` architecture (3 layers, 4 heads, emb_dim=48), 3 passes, seed 123 (1.03 MB).
  - `tiny_ch07.trace.npz`: `tiny_ch07` reference architecture (1 layer, 2 heads, emb_dim=12), 3 passes, seed 123 (144 KB).
- Test suites:
  - `tests/probe/test_hooks_and_capture.py`: Normal and exceptional lifecycle cleanup, module call counting, unbatched CPU storage.
  - `tests/probe/test_reduce.py`: Stride invariants, IndexPlan token windowing, full tensor stats, and NaN/Inf sanitization.
  - `tests/probe/test_builders.py`: Unit tests for all step builders with synthetic and live model tensors.
  - `tests/probe/test_generation.py`: Generation equivalence to `previous_chapters.generate()`, context cropping, and EOS stopping.
  - `tests/probe/test_recorder.py`: Non-interference verification, template order coverage, and NPZ round-tripping.
  - `tests/probe/test_compat_recorder.py`: End-to-end recording across all 8 compatibility matrix variants, verifying logit bit-identity and fidelity.
  - `tests/probe/test_weakness_recorder.py`: Single-token prompt warning, context cropping, attention failure fallback, NaN weight replacement, and determinism.
  - `tests/integration/test_record_cli.py`: Full CLI record invocation and validation of committed sample traces.

### Performance Verification (§8.4 Budgets)
| Metric | Measured | Target Budget | Result |
|---|---|---|---|
| Record `small_bias` (3 passes) | **0.028 s** | < 5.0 s | Passed (178x under budget) |
| Record `gpt2_small_shape` (3 passes, 8 tokens) | **1.972 s** | < 60.0 s | Passed (30x under budget) |
| `gpt2_small_shape` Trace file size | **6.25 MB** | < 50.0 MB | Passed (8x under budget) |
| `gpt2_small_shape` Load & validate time | **0.228 s** | < 2.0 s | Passed (8.7x under budget) |

### Verification & Test Summary
- **Automated tests:** 143 passed in 9.08s (all unit, regression, compatibility, and weakness tests), 145 passed including `slow` tests in 11.5s.
- **Non-interference guarantee:** Verified `torch.equal()` bit-identical logits on CPU with and without recorder, unchanged model `training` and `requires_grad` flags, and empty hook lists across all modules.
- **Fidelity verification:** Verified attention scores and residual additions achieve exact numerical fidelity ($< 10^{-4}$ error).
- **Static checks:** `ruff format --check` (clean across 124 files), `ruff check` (clean), `mypy` (no issues in 89 source files), and import boundary tests clean.

---

## [0.3.0] - 2026-11-08 — Phase 3: Model Loading, Config Inference, Text and Adapters

**Requirements addressed:** R6, R8, R11

### Added
- `src/tilevis/probe/loading/code_import.py`: Dynamic user code importing (`import_model_module`) via unique module names (`tilevis_user_<sha1>`) with isolated `sys.path` modification in try-finally and internal caching.
- `src/tilevis/probe/loading/profile.py`: `ModelProfile` dataclass and TOML profile loader (`load_profile`) resolving relative file paths relative to the profile location.
- `src/tilevis/probe/loading/checkpoint.py`: Safe checkpoint loader (`load_checkpoint`) supporting state dict container unwrapping (`model_state_dict`, `state_dict`, `model`, `nn.Module`), prefix stripping (`module.`, `_orig_mod.`), and secure pickle detection.
- `src/tilevis/probe/loading/config_infer.py`: Model architecture shape inference (`infer_config_from_state_dict`) for `vocab_size`, `emb_dim`, `context_length`, `n_layers`, `qkv_bias`, and provisional `n_heads`, with strict contradiction checking reporting both expected and actual values.
- `src/tilevis/probe/loading/lora.py`: Detection of unmerged LoRA parameters (`detect_lora_tensors`) and exact dense weight folding (`merge_lora_state_dict`, $W_{\text{eff}} = W + \alpha (A B)^T$) with alpha fallback warnings.
- `src/tilevis/probe/loading/factory.py`: `build_model()` assembling ready `nn.Module` instances across custom factory callables, class + checkpoint, and class with seeded random initialization, with non-persistent `.mask` buffer tolerance.
- `src/tilevis/probe/text/tokenizer.py`: `Tokenizer` protocol and `TiktokenTokenizer` GPT-2 BPE implementation with leading space (`·`), newline (`↵`), and undecodable byte (`<0xAB>`) display formatting, plus vocabulary range validation (`validate_token_ids`).
- `src/tilevis/probe/text/templates.py`: `TEMPLATES` registry supporting `raw` identity and `alpaca` instruction tuning formats matching `gpt_instruction_finetuning.py`.
- `src/tilevis/probe/adapters/`: Architecture adapter contract (`ModelSpec`, `BlockSpec`, `ArchitectureAdapter`), `ADAPTERS` registry, `gpt_scratch` adapter with automatic block submodule resolution, `generic` fallback adapter, and `select_adapter` auto-detection.
- `src/tilevis/probe/weights.py`: `WEIGHT_EXTRACTORS` registry extracting weights from `nn.Linear`, `LinearWithLoRA`, `nn.Embedding`, and `LayerNorm`, normalized to the presentation orientation $(D_{\text{in}} \times D_{\text{out}})$.
- `src/tilevis/probe/inspect.py`: Probe inspection utility and CLI subcommand `tilevis inspect` supporting both `--profile` and `--checkpoint` flags.
- `tests/fixtures/tiny_models.py`: Model builder and checkpoint fixture generator for all 8 `compat_matrix` variants in plain, LoRA-wrapped, and DataParallel-wrapped styles.
- Unit and compatibility test suites under `tests/probe/`: `test_code_import.py`, `test_checkpoint.py`, `test_config_infer.py`, `test_lora.py`, `test_text.py`, `test_adapters.py`, `test_weights.py`, `test_inspect_cli.py`, `test_compat_probe.py`.

### Verified Checkpoint Styles
- **Plain state dict:** `tiny_ch07`, `small_bias`, `heads_8_of_64`, `head_dim_1`, `deep_narrow`, `gpt2_small_shape`.
- **LoRA-wrapped state dict:** `lora_small` with rank=4, alpha=8.0 (verified output matches unmerged model within $10^{-5}$).
- **Container / DataParallel wrapped:** `wrapped_ckpt` containing `model_state_dict` and `module.` prefixes.
- **Pickled `nn.Module`:** Verified rejection with guidance when `allow_pickle=False`, and successful unwrapping when `allow_pickle=True`.
- **Custom model variants:** Subclasses with `pos_emb=None` (adapter sets `position_embedding=None`).

### Verification & Test Summary
- **Automated tests:** 108 passed in 6.16s, 90% overall test coverage.
- **Compatibility matrix:** All 8 configurations verified for exact parameter inference, model construction, bit-for-bit CPU logit equivalence (within $10^{-5}$ for LoRA), and adapter confidence $\ge 0.90$.
- **Weakness probes:**
  - Contradictory profile parameters (`n_layers`, `emb_dim`, `vocab_size`) raise `ConfigInferenceError` displaying both values.
  - LoRA checkpoints without explicit `[model.lora.alpha]` log a warning and fall back to 16.0.
  - Vocabulary overflow raises clear `ModelLoadError`.
  - Non-ASCII and Korean paths verified for checkpoint loading.
  - Unpickling full module code rejected safely when `allow_pickle=False`.
  - Name clashes between identical filenames in different directories prevented via unique hashing.
  - `sys.path` restored reliably even on module import failure.
- **Static checks:** `ruff format --check` (clean), `ruff check` (clean), `mypy --strict` on `core`, `narration`, `render` (clean), `mypy` on `probe` (clean), and import boundary tests clean.
- **Manual check:** Executed `tilevis inspect --checkpoint` on a generated `small_bias.pth` model and `tilevis inspect --profile profiles/tiny_random.toml`, verifying clear formatted output.

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
