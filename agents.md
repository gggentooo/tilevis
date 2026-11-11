# agents.md: Instructions for AI Coding Agents

This file is the **authoritative guide** for any AI agent (or human acting like one) that implements the **LLM Tile Visualizer** (`tilevis`).
- Read it fully before writing code.
- Follow it strictly.
- If something here conflicts with `architecture_proposal.md`, **this file wins**. The proposal was an early draft, and this file includes the user's later decisions.

Companion documents:

| File | Audience | Purpose |
|---|---|---|
| `agents.md` (this file) | AI agents | Rules, libraries, structure, implementation guide, phased plan |
| `technical_document.md` | Human contributors | Technical reference: setup, structure, formats, extension how-tos |
| `changelog.md` | Everyone | Phase-by-phase record of what was built, tested and decided |
| `architecture_proposal.md` | Historical | Original proposal (superseded where it differs from this file) |

---

## 1. Project summary and requirements

`tilevis` is a Python program that turns a GPT-style LLM's forward pass into a **step-by-step, tile-based animation**.
- **Tile meaning:** each colored tile is one number: an activation value or a weight.
- **Audience:** a school presentation for students with little or no LLM background.
- **Tone:** visuals must be intuitive and fun, without too much detail.

### 1.1 Requirements (numbered so they can be referenced from code, tests and the changelog)

| ID | Requirement |
|---|---|
| R1 | Show the **weights** and **values** (activations) of the network as **colored tiles**. |
| R2 | The presenter **advances manually**. **One key press computes one whole layer** (one step). Stepping backwards must also work. |
| R3 | Show **how input vectors are transformed by each layer's weights**, for **each forward pass**. |
| R4 | Show **both** the internal layers **and** the final **next-word prediction**. |
| R5 | Be intuitive and fun for non-experts. Advanced detail is hidden by default and available on demand ("nerd mode"). |
| R6 | Sit **on top of** `ch07/01_main-chapter-code/previous_chapters.py` **without modifying it**. |
| R7 | Two main components: a **probe** module (hooks into the model and watches values) and a **render** module (draws the values to the viewport). |
| R8 | **Model-agnostic.** Teammates built many models with different `n_layers`, `emb_dim`, `n_heads`, `context_length`, `qkv_bias`, and possibly LoRA. The probe must **adapt dynamically**, with no hard-coded sizes. |
| R9 | Renderer library: **pygame** (the `pygame-ce` distribution, see §3). |
| R10 | Runs on the **Windows lecture-hall PC**: one screen shared via screenshare, with keyboard and mouse input only. |
| R11 | Modular, expandable and maintainable long-term. |

### 1.2 Key consequences of the requirements

- **Record, then play.** The probe records a forward-generation run into a **Trace file**. The renderer plays the Trace back. Playback needs only `numpy` and `pygame-ce`, with no PyTorch. This makes the lecture-hall setup trivial and robust (R10).
- **Adapters plus config inference** make the probe work with any teammate's checkpoint (R8).
- **Playlist filtering** lets the presenter pick a detail level. One press always equals one step, so deep models stay presentable (R2, R5).

---

## 2. Ground rules (non-negotiable)

1. **Never modify anything under `ch07/`.** Treat it as read-only third-party code. Load it by file path (§6.1.1).
2. **Respect the module dependency rules in §4.3.** They are enforced by a test.
   - `core` and `narration` must never import `torch` or `pygame`.
   - `render` must never import `torch`.
   - `probe` must never import `pygame`.
3. **The Trace format is the only interface between probe and render.** Any change to it requires:
   - a `TRACE_SCHEMA_VERSION` bump,
   - a migration (or a clear rejection message),
   - updated tests,
   - an entry in `changelog.md` marked **BREAKING**.
4. **No hard-coded model sizes** anywhere: layer count, dimensions, heads, vocab or context. Everything comes from the `ModelSpec` (§5.2).
5. **Hooks are read-only.** They must never alter tensors, model state, `requires_grad`, or the model's train/eval mode after recording. A test enforces bit-identical logits with and without hooks.
6. **Every phase ends with its test step passing** (§9), and with `changelog.md` updated (§10), before the next phase starts.
7. **Cross-platform by construction:**
   - Use `pathlib` for paths.
   - Pass `encoding="utf-8"` explicitly for every text read and write. Korean Windows defaults to cp949.
   - Never use shell-specific behavior.
8. **No new runtime dependency** without justification. If one is added, update §3, `technical_document.md` and `changelog.md`.
9. **Prefer registries over `if/elif` chains** for anything that will grow: step builders, scenes, colormaps, reducers, adapters, templates, themes and weight extractors.
10. **Use `logging`, not `print`.** The exception is the CLI's user-facing output.
11. **Keep files focused.** As a guideline, stay under about 400 lines per module, with one responsibility per module.
12. **When unsure about a user-facing decision, ask the user.** Do not guess. Record the decision in the changelog.

---

## 3. Libraries

### 3.1 Runtime

| Library | Version | Used by | Purpose | Why this choice |
|---|---|---|---|---|
| Python | **3.11 – 3.13** | all | Language | 3.11 gives stdlib `tomllib`, faster CPython and modern typing. 3.12 is the reference dev version. |
| `numpy` | `>=1.26` | core, probe, render | Arrays in Traces; colormap LUTs | Universal. The Trace stores numpy arrays. |
| `pygame-ce` | `>=2.5` | render | Window, drawing, input, fonts | Community Edition of pygame. It is **import-compatible** (`import pygame`), more actively maintained, ships Windows wheels quickly for new Python versions, and adds useful APIs. **Never install `pygame` and `pygame-ce` together** because they conflict. |

### 3.2 Recording extra (`pip install -e ".[record]"`)

| Library | Version | Used by | Purpose | Notes |
|---|---|---|---|---|
| `torch` | `>=2.3` | probe | Runs the model, forward hooks | The project README uses 2.3.1. **torch 2.3.x needs `numpy<2`.** torch ≥ 2.4 works with numpy 2. Use `torch.load(..., weights_only=True)` by default. |
| `tiktoken` | `>=0.7` | probe | GPT-2 BPE tokenizer | Same tokenizer as the original project. |
| `matplotlib` | `>=3.8` | (indirect) | Required only because `previous_chapters.py` imports it at module level | `tilevis` itself must never import matplotlib. |

### 3.3 Development extra (`pip install -e ".[dev]"`)

| Library | Purpose |
|---|---|
| `pytest`, `pytest-cov`, `pytest-timeout` | Tests, coverage, hang protection |
| `hypothesis` | Property-based tests (colormap, reducers, serialization, player navigation) |
| `ruff` | Linting and formatting (replaces flake8, isort and black) |
| `mypy` | Static typing (strict for `core`, `narration`, `render`) |

### 3.4 Build extra (`pip install -e ".[build]"`, Phase 8 only)

| Library | Purpose |
|---|---|
| `pyinstaller>=6` | Optional single-folder Windows build of the **player** (no Python needed on the lecture PC). It must be built **on Windows** because PyInstaller does not cross-compile. |

### 3.5 Explicitly rejected (do not introduce)

| Library | Reason |
|---|---|
| `matplotlib` inside `tilevis` | Too slow for interactive animation. The colormaps are implemented as numpy LUTs. |
| `pydantic`, `attrs` | Stdlib `dataclasses` are enough, and they keep dependencies minimal. |
| `click`, `typer` | Stdlib `argparse` is enough for 5 subcommands. |
| `PyYAML` | TOML (`tomllib`, stdlib) is used for all config files. |
| Threads, multiprocessing | Not needed. Record and play are separate phases, and pygame must own the main thread. |
| `tensorflow` | Only needed by `gpt_download.py`. That path is out of scope (see §12). |

---

## 4. Architecture and module structure

### 4.1 Big picture

```
                 RECORD TIME (dev laptop, needs torch)              PLAY TIME (lecture PC, no torch)
┌──────────────────────┐   ┌──────────────────────────────┐        ┌───────────────────────────────┐
│ previous_chapters.py │──►│ probe                        │        │ render                        │
│ (+ teammate code,    │   │  loading → adapters → hooks  │        │  player/playlist → scenes →   │
│  checkpoints .pth)   │   │  → builders → recorder       │        │  widgets → pygame window      │
└──────────────────────┘   └──────────────┬───────────────┘        └───────────────▲───────────────┘
                                          │ Trace (core)                           │ Trace (core)
                                          ▼                                        │
                                  traces/*.trace.npz  ─────── copy / USB ──────────┘
                                          ▲
                         narration (captions) is used by render
```

### 4.2 Repository layout (target state)

```
llm_tile_visualizer/
├── agents.md                    # this file
├── technical_document.md
├── changelog.md
├── architecture_proposal.md     # historical
├── README.md                    # short quick start (Phase 8)
├── pyproject.toml               # packaging, deps, extras, tool config (ruff, mypy, pytest)
├── .gitignore                   # ignores *.pth, traces/*.npz (except samples), venvs, caches
├── .github/workflows/ci.yml     # optional CI: ubuntu/windows/macos × py3.11/3.12
├── ch07/                        # READ-ONLY original project
├── profiles/                    # model profiles (TOML), one per teammate model
│   ├── tiny_random.toml         # no checkpoint; random tiny model for demos/tests
│   └── example_team_model.toml  # documented template
├── presenter/
│   └── default.toml             # presenter settings (theme, detail level, keys…)
├── traces/
│   └── samples/                 # small committed sample traces for playback-only machines
├── scripts/
│   ├── play_demo.bat            # Windows double-click launcher
│   └── play_demo.sh
├── src/tilevis/
│   ├── __init__.py              # __version__
│   ├── __main__.py              # `python -m tilevis` → cli.main()
│   ├── cli.py                   # argparse subcommands; lazy-imports probe/render
│   ├── logging_setup.py
│   ├── core/                    # SHARED CONTRACT: stdlib + numpy only
│   │   ├── kinds.py             # StepKind, Semantic, DetailLevel, Stage, TensorRole
│   │   ├── trace.py             # Trace, PassRecord, Step, TensorSnapshot, Prediction, ModelSpecInfo
│   │   ├── serialization.py     # save_trace / load_trace (.trace.npz), schema migrations
│   │   ├── validation.py        # validate_trace() – structural & numeric invariants
│   │   ├── registry.py          # generic Registry[K, V] with decorator registration
│   │   ├── config.py            # dataclass settings + TOML loading helpers
│   │   └── errors.py            # TilevisError hierarchy
│   ├── probe/                   # MODULE A: torch-aware, never imports pygame
│   │   ├── loading/
│   │   │   ├── code_import.py   # import user model code from a file path (isolated)
│   │   │   ├── checkpoint.py    # load/unwrap state dicts, strip prefixes
│   │   │   ├── config_infer.py  # infer GPT config from state-dict shapes
│   │   │   ├── lora.py          # detect & merge LoRA weights
│   │   │   ├── profile.py       # ModelProfile dataclass + TOML parsing
│   │   │   └── factory.py       # build a ready nn.Module from a profile
│   │   ├── text/
│   │   │   ├── tokenizer.py     # Tokenizer protocol + TiktokenTokenizer
│   │   │   └── templates.py     # prompt templates registry ("raw", "alpaca")
│   │   ├── adapters/
│   │   │   ├── base.py          # ArchitectureAdapter protocol, ModelSpec, BlockSpec
│   │   │   ├── gpt_scratch.py   # adapter for previous_chapters-style GPTModel (duck-typed)
│   │   │   └── generic.py       # fallback: any nn.Module, leaf modules in call order
│   │   ├── weights.py           # WeightExtractor registry (Linear, LoRA wrapper, Embedding, LayerNorm)
│   │   ├── hooks.py             # HookManager context manager
│   │   ├── capture.py           # CaptureBuffer for one forward pass
│   │   ├── reduce.py            # reducers + IndexPlan (tensor → displayable snapshot)
│   │   ├── builders/            # Step builders (registry keyed by Semantic)
│   │   │   ├── base.py
│   │   │   ├── embedding.py
│   │   │   ├── layernorm.py
│   │   │   ├── attention.py     # recompute + fidelity check + n_heads detection
│   │   │   ├── feedforward.py
│   │   │   ├── residual.py
│   │   │   ├── block_summary.py
│   │   │   ├── output.py        # final norm, LM head, next-token prediction
│   │   │   └── generic.py
│   │   ├── generation.py        # multi-pass generation loop (greedy / sampling)
│   │   └── recorder.py          # TraceRecorder – orchestrates everything above
│   ├── narration/               # stdlib + core only
│   │   ├── captions.py          # CaptionProvider: Step → Caption(title, body, nerd)
│   │   └── locales/
│   │       └── en.toml          # caption texts keyed by Semantic/StepKind
│   ├── render/                  # MODULE B: pygame + numpy, never imports torch
│   │   ├── app.py               # App: owns window, main loop, tick(dt, events)
│   │   ├── platform.py          # Windows DPI awareness, display queries
│   │   ├── window.py            # windowed / borderless-fullscreen / exclusive modes
│   │   ├── commands.py          # Command enum (NEXT, PREV, NEXT_STAGE, …)
│   │   ├── input_map.py         # pygame events → Commands (configurable bindings)
│   │   ├── playlist.py          # Trace + PresenterSettings → ordered list of StepRefs
│   │   ├── player.py            # StepPlayer: cursor over playlist, navigation
│   │   ├── layout.py            # resolution-independent regions & UiScale
│   │   ├── colormap.py          # Colormap registry, vectorized LUT mapping
│   │   ├── normalization.py     # color scaling policies (per-tensor, shared, global)
│   │   ├── tiles.py             # TileGrid: array → cached Surface, hit testing
│   │   ├── text.py              # FontManager (bundled fonts, size cache)
│   │   ├── animation.py         # Tween, Timeline, easing functions
│   │   ├── scenes/
│   │   │   ├── base.py          # Scene protocol + SceneContext
│   │   │   ├── registry.py      # StepKind → Scene class, with GenericScene fallback
│   │   │   ├── generic.py
│   │   │   ├── title.py         # title & end cards
│   │   │   ├── tokenize.py
│   │   │   ├── embedding.py
│   │   │   ├── layernorm.py
│   │   │   ├── matmul.py        # THE core scene: input × weights → output
│   │   │   ├── qkv.py
│   │   │   ├── attention.py     # attention pattern (heatmap + arcs)
│   │   │   ├── attention_mix.py
│   │   │   ├── activation.py
│   │   │   ├── residual.py
│   │   │   ├── block_summary.py
│   │   │   ├── logits.py
│   │   │   └── prediction.py    # next-word bar chart + token appended
│   │   ├── widgets/
│   │   │   ├── caption_box.py
│   │   │   ├── hud.py           # pass counter, step counter
│   │   │   ├── minimap.py       # pipeline overview, clickable
│   │   │   ├── legend.py        # color bar
│   │   │   ├── buttons.py       # on-screen ◀ ▶ buttons
│   │   │   ├── tooltip.py
│   │   │   ├── nerd_panel.py
│   │   │   └── help_overlay.py
│   │   └── themes/
│   │       ├── base.py          # Theme dataclass
│   │       └── builtin.py       # "dark", "light" (both colorblind-safe)
│   └── assets/
│       └── fonts/               # bundled OFL font (e.g. Noto Sans) + OFL.txt
└── tests/
    ├── conftest.py              # headless SDL env, shared fixtures
    ├── compat_matrix.py         # model configs every phase tests against (§8.3)
    ├── fixtures/
    │   ├── fake_traces.py       # torch-free synthetic Trace factory
    │   └── tiny_models.py       # builds/saves tiny GPTModels (requires torch)
    ├── core/  probe/  narration/  render/  integration/
    └── test_import_boundaries.py
```

### 4.3 Dependency rules (enforced by `tests/test_import_boundaries.py`)

```
core       → stdlib, numpy
narration  → core
probe      → core, torch, tiktoken, user model code   (NEVER pygame, NEVER render/narration)
render     → core, narration, pygame, numpy           (NEVER torch, NEVER probe)
cli        → all, but probe/render are imported lazily inside the subcommand functions
```

How the boundary test works:
- It parses every `.py` file under each package with `ast` and asserts the allowed import set.
- It also runs, in a subprocess, `import tilevis.render, tilevis.narration, tilevis.core; tilevis.cli` with the `play` path. Then it asserts `"torch" not in sys.modules`.

### 4.4 Extension points (registries)

| Registry | Key | Lives in | Add a new… |
|---|---|---|---|
| `ADAPTERS` | adapter name | `probe/adapters` | model architecture family |
| `WEIGHT_EXTRACTORS` | module type / predicate | `probe/weights.py` | layer type whose weights should be shown |
| `STEP_BUILDERS` | `Semantic` | `probe/builders` | kind of recorded step |
| `REDUCERS` | strategy name | `probe/reduce.py` | way to shrink big tensors |
| `TEMPLATES` | template name | `probe/text/templates.py` | prompt format |
| `SCENES` | `StepKind` | `render/scenes/registry.py` | visualization for a step kind |
| `COLORMAPS` | name | `render/colormap.py` | color scheme |
| `THEMES` | name | `render/themes/builtin.py` | visual theme |
| `NORMALIZERS` | name | `render/normalization.py` | color scaling policy |
| caption locales | language code | `narration/locales/*.toml` | caption language |

`core/registry.py` provides one generic implementation:

```python
class Registry(Generic[K, V]):
    def register(self, key: K) -> Callable[[V], V]: ...   # decorator
    def get(self, key: K, default: V | None = None) -> V: ...
    def keys(self) -> list[K]: ...
```

Registering an existing key raises an error. `override=True` is allowed explicitly for user plugins.

---

## 5. Data contract (`tilevis.core`)

### 5.1 Vocabulary (`core/kinds.py`)

All of these are `StrEnum`s, stored as plain strings in the Trace for forward compatibility. Unknown strings found when loading a newer trace must **not crash**. They map to `GENERIC` behavior.

- **`StepKind`**: *how to draw* a step. Scenes are keyed by this.
  - `PROMPT`, `TOKENIZE`, `EMBEDDING`, `LAYERNORM`, `LINEAR`, `QKV_PROJECTION`
  - `ATTENTION_PATTERN`, `ATTENTION_MIX`, `ACTIVATION`, `RESIDUAL_ADD`
  - `BLOCK_SUMMARY`, `LOGITS`, `NEXT_TOKEN`, `GENERIC`
- **`Semantic`**: *what a step means*. Captions are keyed by this.
  - `prompt`, `tokenize`, `embed`
  - `block.norm1`, `block.att.qkv`, `block.att.scores`, `block.att.mix`, `block.att.out`
  - `block.resid1`, `block.norm2`, `block.ff.expand`, `block.ff.act`, `block.ff.contract`, `block.resid2`
  - `block.summary`, `final_norm`, `lm_head`, `next_token`, `generic`
- **`DetailLevel`** (int): `STORY = 1`, `STANDARD = 2`, `DEEP = 3`.
- **`Stage`**: `INPUT`, `BLOCK`, `OUTPUT`. Used by the minimap and the "next stage" navigation.
- **`TensorRole`**:
  - `input`, `output`, `weight`, `bias`, `scale`, `shift`, `shortcut`, `update`
  - `queries`, `keys`, `values`, `attn_weights`, `token_vectors`, `position_vectors`
  - `logits`, `probs`, `extra`

### 5.2 Dataclasses (`core/trace.py`)

All dataclasses are `@dataclass(frozen=True, slots=True)`. Arrays are made read-only (`arr.setflags(write=False)`) on construction.

```python
TRACE_SCHEMA_VERSION = 1

@dataclass(frozen=True, slots=True)
class AxisInfo:
    name: str                    # "token" | "model_dim" | "ff_dim" | "head_dim" | "vocab" | "head"
    full_size: int               # size before reduction
    indices: np.ndarray          # int64; which original indices are shown (len == displayed size)
    labels: tuple[str, ...] | None  # e.g. token strings for the token axis, words for vocab axis

@dataclass(frozen=True, slots=True)
class TensorStats:
    min: float; max: float; mean: float; std: float
    abs_max: float; abs_p99: float   # computed on the FULL tensor (before reduction)

@dataclass(frozen=True, slots=True)
class TensorSnapshot:
    name: str                    # human/debug name, e.g. "trf_blocks.0.att.W_query.weight"
    role: str                    # TensorRole value
    data: np.ndarray             # float32, 1-D or 2-D (3-D allowed only for attn heads: head×q×k)
    axes: tuple[AxisInfo, ...]   # one per data dimension
    full_shape: tuple[int, ...]  # original tensor shape (batch dim removed)
    reduction: str               # reducer name used ("none", "stride", ...)
    stats: TensorStats

@dataclass(frozen=True, slots=True)
class Fidelity:
    status: str                  # "exact" | "approximate" | "unverified" | "failed"
    max_abs_err: float | None
    note: str = ""

@dataclass(frozen=True, slots=True)
class Step:
    id: str                      # unique within trace: "p0.b3.att.scores"
    kind: str                    # StepKind
    semantic: str                # Semantic
    stage: str                   # Stage
    block_index: int | None
    detail_level: int            # DetailLevel: minimum level at which the step is shown
    module_path: str | None      # e.g. "trf_blocks.3.att"
    tensors: tuple[TensorSnapshot, ...]
    meta: Mapping[str, JSONValue]   # JSON-serializable only (n_heads, eps, activation name…)
    fidelity: Fidelity

@dataclass(frozen=True, slots=True)
class Prediction:
    candidate_ids: tuple[int, ...]       # top-k token ids, descending probability
    candidate_tokens: tuple[str, ...]
    candidate_probs: tuple[float, ...]
    chosen_id: int
    chosen_token: str
    strategy: str                        # "greedy" | "sample"

@dataclass(frozen=True, slots=True)
class PassRecord:
    index: int
    input_ids: tuple[int, ...]           # the (possibly context-cropped) model input
    input_tokens: tuple[str, ...]        # display strings per token
    text_before: str
    steps: tuple[Step, ...]
    prediction: Prediction | None        # None only if generation stopped (EOS)

@dataclass(frozen=True, slots=True)
class ModelSpecInfo:                     # serializable summary of probe.adapters.ModelSpec
    name: str; adapter: str
    n_layers: int; emb_dim: int; n_heads: int | None
    vocab_size: int; context_length: int | None
    n_params: int; extras: Mapping[str, JSONValue]

@dataclass(frozen=True, slots=True)
class Trace:
    schema_version: int
    tilevis_version: str
    created_at: str                      # ISO-8601
    model: ModelSpecInfo
    prompt: str
    settings: Mapping[str, JSONValue]    # generation + reduction settings used
    passes: tuple[PassRecord, ...]
```

**Orientation convention (document it in docstrings):** every weight matrix is stored as **(in × out)**. Activations are stored as **(token × feature)**. So `output_row = input_row @ weight` holds visually. `nn.Linear.weight` is `(out, in)` and must be transposed by the weight extractor.

### 5.3 Trace file format (`core/serialization.py`)

- **Extension:** `*.trace.npz`. It is a standard numpy `.npz` (zip) archive.
- **Manifest:** key `__manifest__` is a `uint8` array containing UTF-8 JSON. It holds every dataclass field except arrays. Each array field is replaced by a reference string `"@a<N>"`.
- **Arrays:** keys `a0`, `a1`, … hold the arrays.
- **Saving:** use `np.savez_compressed`.
- **Loading:** use `np.load(path, allow_pickle=False)`. **Pickle is never allowed.**
- **Versioning:**
  - `load_trace` reads `schema_version`.
  - If it is lower, apply the migrations in `MIGRATIONS[v] : dict -> dict` in sequence.
  - If it is higher, raise `TraceFormatError("trace was made by a newer tilevis; please update")`.
- **Validation:** `load_trace` always calls `validate_trace()` (§5.4). `save_trace` validates before writing.
- **Atomic writes:** write to `path.with_suffix(".tmp")`, then `os.replace`.

### 5.4 Invariants checked by `validate_trace()`

- Step ids are unique. Every `kind`, `semantic` and `stage` string is non-empty.
- Every `TensorSnapshot`:
  - `data.ndim == len(axes)`
  - `len(axes[i].indices) == data.shape[i]`
  - indices are within `[0, full_size)` and strictly increasing
  - `data` is finite (NaN/Inf are replaced by the probe and flagged in `meta`, never stored)
- For the `token` axis, `labels` exist and their length matches.
- Each pass has at most one `NEXT_TOKEN` step, and it is the pass's last step.
- `Prediction.candidate_probs` are descending and within `[0, 1]`.
- `block_index` is in `[0, model.n_layers)` when it is set.

---

## 6. Implementation guide

### 6.1 Probe (Module A)

#### 6.1.1 Loading user model code (`loading/code_import.py`)

- Import the model module by **file path** with `importlib.util.spec_from_file_location`.
  - Use a unique module name: `tilevis_user_<sha1(path)[:8]>`.
  - This avoids `sys.path` pollution and name clashes between teammates' files.
- While executing the module, temporarily prepend its parent directory to `sys.path`, so sibling imports (for example `gpt_download`) work. Restore `sys.path` in a `finally`.
- Cache imported modules by resolved path.
- The default code path is `ch07/01_main-chapter-code/previous_chapters.py`, resolved relative to the repository root. A profile may override it.

#### 6.1.2 Model profiles (`loading/profile.py`)

A profile is a TOML file describing *one* model. Relative paths are resolved **relative to the profile file**.

```toml
[profile]
name = "Team A – 3-layer instruction model"

[model]
code = "../ch07/01_main-chapter-code/previous_chapters.py"
class = "GPTModel"            # constructed as Class(cfg_dict)
checkpoint = "../models/team_a.pth"   # omit → random init (seeded)
# factory = "path/to/file.py:build_model"   # alternative: callable returning nn.Module
allow_pickle = false          # true only for full-module pickles; logs a security warning
seed = 123                    # used for random init and sampling

[model.config]                # OPTIONAL overrides; anything missing is inferred
n_heads = 4

[model.lora]                  # only needed for checkpoints containing LoRA tensors
alpha = 16

[text]
tokenizer = "gpt2"
template = "raw"              # "raw" | "alpaca"

[adapter]
name = "auto"                 # "auto" | "gpt_scratch" | "generic"
[adapter.roles]               # OPTIONAL module-path overrides for renamed modules
# blocks = "trf_blocks"
```

Profiles are parsed into a frozen `ModelProfile` dataclass. Unknown keys produce a warning, not an error, for forward compatibility.

#### 6.1.3 Checkpoint loading (`loading/checkpoint.py`)

- `torch.load(path, map_location="cpu", weights_only=not allow_pickle)`.
- **Unwrap** common containers: if the result is a dict with a `model_state_dict` / `state_dict` / `model` key holding a mapping, use that. This covers ch05-style checkpoints that also store the optimizer.
- If the result is an `nn.Module` (pickled full model), take its `state_dict()` and keep the module.
- **Strip prefixes** `module.` (DataParallel) and `_orig_mod.` (`torch.compile`).
- Return a `LoadedCheckpoint(state_dict, module_or_none, source_path)`.

#### 6.1.4 Config inference (`loading/config_infer.py`)

Infer a `GPTModel` cfg dict from state-dict **shapes**:

| Field | Rule |
|---|---|
| `vocab_size`, `emb_dim` | `tok_emb.weight.shape` |
| `context_length` | `pos_emb.weight.shape[0]` |
| `n_layers` | count distinct `i` in keys `trf_blocks.{i}.` |
| `qkv_bias` | `"trf_blocks.0.att.W_query.bias"` (or `.linear.bias` for LoRA) in keys |
| `drop_rate` | always `0.0` (eval only) |
| `n_heads` | **not inferable from shapes**. Priority: profile/CLI value, then **automatic detection** at record time (§6.1.9), then a provisional value for construction. |

Merge order: **profile `[model.config]` > CLI overrides > inferred > defaults.** If an explicit value contradicts the checkpoint shape (for example the profile says `n_layers = 4` but the checkpoint has 6), raise `ConfigInferenceError` with a clear message. `n_heads` does not change parameter shapes in `previous_chapters.MultiHeadAttention`, so a provisional value is safe for construction. The attention builder then fixes it.

#### 6.1.5 LoRA support (`loading/lora.py`)

The teammates' LoRA code (`exercise_experiments.py`) wraps `nn.Linear` as `LinearWithLoRA(linear, lora)`, where `lora.A` is `(in, r)`, `lora.B` is `(r, out)`, and `forward = linear(x) + alpha * x @ A @ B`.
- **Detect** keys ending in `.lora.A` / `.lora.B`.
- **Merge** into plain weights: `W_eff = W + alpha * (A @ B).T`. Rename `X.linear.weight` → `X.weight` and `X.linear.bias` → `X.bias`, then drop the LoRA keys.
- `alpha` is **not stored** in checkpoints. Take it from `[model.lora].alpha`. If it is missing, use 16 (the exercise default) and log a **warning** that predictions may differ.
- For live modules that still contain LoRA wrappers (factory path), the `WEIGHT_EXTRACTORS` entry for "has `.linear` and `.lora`" returns the effective weight, and hooks attach to the wrapper.

#### 6.1.6 Model factory (`loading/factory.py`)

`build_model(profile, overrides) -> BuiltModel(model, cfg, tokenizer, template, notes)`:
1. Import the code. Then use one of three paths:
   - **(a)** with `factory`: call it,
   - **(b)** with `class` and a checkpoint: infer the cfg, merge LoRA, construct, and `load_state_dict(strict=True)`,
   - **(c)** with `class` and no checkpoint: seed, then construct with the given or default tiny cfg.
2. On missing or unexpected keys, raise `ModelLoadError`.
   - List up to 10 keys.
   - Add hints, for example "looks like a LoRA checkpoint; set [model.lora]" or "n_layers mismatch".
   - The only tolerated difference is non-persistent/derivable buffers whose key ends in `.mask`.
3. Then `model.eval()`, `model.requires_grad_(False)`, and move to the device (`cpu` by default, `--device auto|cpu|cuda|mps`).

#### 6.1.7 Text (`text/`)

- `Tokenizer` protocol:
  - `encode(str) -> list[int]`
  - `decode(list[int]) -> str`
  - `token_display(id) -> str`
  - `eos_id: int | None`
  - `name`
- `TiktokenTokenizer("gpt2")`:
  - `eos_id = 50256`
  - Uses `allowed_special={"<|endoftext|>"}` exactly like `previous_chapters.text_to_token_ids`.
- `token_display`:
  - Decode a single id with `errors="replace"`.
  - Show a leading space as `·` and newlines as `↵`.
  - Show undecodable partial UTF-8 bytes (common for Korean text in the GPT-2 BPE) as `<0xAB>`.
- Templates registry:
  - `raw` (identity)
  - `alpaca`: the same text as `format_input` in `gpt_instruction_finetuning.py`, with an empty input section. It ends with `"\n\n### Response:\n"`.
- **Validation:**
  - Every token id must be `< vocab_size`. Otherwise raise a `ModelLoadError` explaining the tokenizer/model mismatch.
  - Prompt length plus `new_tokens` beyond `context_length` crops from the left, exactly like `generate()` (`idx[:, -context_size:]`). This is noted in the pass `meta`.

#### 6.1.8 Adapters (`adapters/`)

An adapter maps semantic **roles** to concrete submodules, so the rest of the probe never touches attribute names.

```python
@dataclass(frozen=True)
class BlockSpec:
    index: int
    path: str                         # "trf_blocks.0"
    norm1: str | None; attention: str | None
    q: str | None; k: str | None; v: str | None; out_proj: str | None
    norm2: str | None; ff: str | None
    ff_in: str | None; ff_act: str | None; ff_out: str | None

@dataclass(frozen=True)
class ModelSpec:
    adapter: str
    token_embedding: str; position_embedding: str | None
    blocks: tuple[BlockSpec, ...]
    final_norm: str | None; lm_head: str
    emb_dim: int; vocab_size: int; context_length: int | None
    n_heads: int | None               # filled/confirmed by attention builder
    residual_style: str               # "pre_norm" (GPT-2) | "unknown"

class ArchitectureAdapter(Protocol):
    name: str
    def matches(self, model: nn.Module) -> float: ...     # confidence 0..1
    def describe(self, model: nn.Module, overrides: Mapping[str, str]) -> ModelSpec: ...
```

- **`gpt_scratch`**: duck-typed check for:
  - `tok_emb`, `pos_emb`, an iterable `trf_blocks`, `final_norm`, `out_head`
  - in each block: `att.{W_query, W_key, W_value, out_proj}`, `ff`, `norm1`, `norm2`
  - `ff_in` / `ff_act` / `ff_out` are found by scanning `ff`'s children in order: the first Linear-like, the first activation-like, the last Linear-like.
  - It works for any `n_layers` and dims. Missing optional roles (for example no `pos_emb`) are `None`, and those steps are skipped.
- **`generic`**: the fallback for unknown architectures.
  - It produces no blocks, only `GENERIC` / `LINEAR` / `LAYERNORM` / `ACTIVATION` steps for each *leaf* module, in the order it was called.
  - Leaves are classified by the `WEIGHT_EXTRACTORS` and a module classifier.
  - The final `lm_head` is the last Linear whose `out_features == vocab_size`.
- `auto` picks the adapter with the highest `matches()` score. It must be ≥ 0.5, else `generic`. Log the choice.

#### 6.1.9 Hooks, capture and builders

- **`HookManager`** (a context manager):
  - Registers `register_forward_pre_hook` / `register_forward_hook` on the module paths requested by builders.
  - Keeps per-module **call counters**, because a module may run more than once per forward. For example `drop_resid` runs twice per block.
  - Removes **all** handles in `__exit__`, even on exceptions.
  - Restores the original `training` flag.
- **`CaptureBuffer`**: `{(module_path, call_idx, "in"|"out"): tensor}`.
  - Stores `t.detach().to("cpu", torch.float32).clone()[0]`, dropping batch dim 0 (batch size is always 1).
  - Only the paths requested by the active builders are stored.
  - Memory is bounded by `O(n_layers × T × 4·emb_dim)`. With demo prompts (T ≤ 64) this is a few MB even for GPT-2 XL.
- **Builders** run **after** each forward pass. Each takes `(ModelSpec, CaptureBuffer, model, IndexPlan, ctx)` and returns `list[Step]`. They are registered by `Semantic`, and the recorder calls them in a fixed **template order** (§6.1.11), not in hook firing order. This makes output deterministic.
- **Attention builder** (the most important one):
  1. Take `x = norm1 output`, and `q, k, v` = the outputs of the `q` / `k` / `v` modules, plus `ctx_true` = the **input** of `out_proj`.
  2. For a candidate `h`, recompute the heads exactly as `MultiHeadAttention.forward` does:
     - causal mask
     - scale by `head_dim ** 0.5`
     - softmax
     - `@ v`
     - merge heads
  3. **Fidelity:** compare the recomputed context to `ctx_true`.
     - `max_abs_err ≤ 1e-4 · max(1, |ctx_true|.max())` → `exact`
     - `≤ 1e-2` relative → `approximate`
     - otherwise `failed`
  4. **n_heads detection:**
     - If `n_heads` is known, verify it.
     - Otherwise try every divisor of `emb_dim`: first `emb_dim // 64` (the GPT-2 convention), then ascending. Choose the **unique** passing candidate.
     - **Weakness:** with `T == 1` every candidate passes, because softmax over one key is 1. In that case keep the provisional value, mark fidelity `unverified`, and log a warning. The CLI should ask for ≥ 2 prompt tokens.
     - Store the result in `ModelSpec` and reuse it for later passes and blocks.
  5. On `failed`, emit a single `GENERIC` "attention (black box)" step: input → output, with a note in `meta`. Never crash.
  6. Emitted steps:
     - `QKV_PROJECTION` (x, Wq, Wk, Wv → Q, K, V)
     - `ATTENTION_PATTERN` (attn weights, head × T × T, plus the mean over heads)
     - `ATTENTION_MIX` (weights × V → context)
     - `LINEAR` (context × W_out → att output, semantic `block.att.out`)
- **Residual builder:**
  - `shortcut = block input`, captured with a pre-hook on the block.
  - `update = attention module output`.
  - `out = norm2 input`.
  - Verify `out ≈ shortcut + update`, and record fidelity. The same applies to the second residual, using the FF output and the block output.
  - If `residual_style != "pre_norm"` or verification fails, emit a `GENERIC` step instead.
- **Block summary builder:**
  - Takes the block input and block output, plus `change = output − input`.
  - Its level is `STORY`, and it is always recorded. Whether it shows is decided by the playlist (§6.2.3).
- **Output builder:**
  - `final_norm` → `LAYERNORM` step.
  - `lm_head` → `LOGITS` step. Use the **last token's** vector only. The weight is reduced to the top-k candidate rows of `out_head` (vocab axis labels = the candidate words).
  - `NEXT_TOKEN` step: probs for the top-k plus the chosen token.
  - The probabilities come from the **real model logits** returned by `forward`, not from hooks. The hooked LM-head result must match them (fidelity check).

#### 6.1.10 Reduction (`reduce.py`)

Big tensors must become displayable **consistently**: a tile must refer to the same original dimension in every step.

- **`IndexPlan`** is built once per recording from the `ModelSpec`. It holds one fixed `indices` array per axis family:
  - `model_dim` (≤ `max_features`, default **64**)
  - `ff_dim` (default **64**)
  - `head_dim`
  - `token` (≤ `max_tokens`, default **24**; keep the **last** N tokens, because the newest token is the one being predicted)
  - `heads` (≤ `max_heads`, default **12**)
- **Reducers** (registry): `none` (size ≤ limit), `stride` (default: evenly spaced indices via `np.linspace(...).round()`, deduplicated), `block_mean`, `max_abs`. Weight matrices reuse the input-axis plan for rows and the output-axis plan for columns.
- `TensorStats` are always computed on the **full** tensor before reduction.
- Replace non-finite values with 0 and set `meta["had_non_finite"] = True`.

#### 6.1.11 Recorder and generation

- `TraceRecorder(built_model, settings).record(prompt) -> Trace`. `RecordSettings` holds:
  - `new_tokens` (default 3)
  - `strategy` (`greedy` | `sample`), `temperature`, `top_k`, `seed`
  - `top_k_display` (default 10)
  - the reduction limits
  - `device`
- **Per-pass template order:**
  1. `prompt` (`PROMPT`, level 1)
  2. `tokenize` (`TOKENIZE`, level 1)
  3. `embed` (`EMBEDDING`, level 1). It holds the token vectors, the position vectors and their sum.
  4. For each block `b`:
     - `block.summary` (level 1)
     - `norm1`, `att.qkv`, `att.scores`, `att.mix`, `att.out`, `resid1` (level 2)
     - `norm2`, `ff.expand`, `ff.act`, `ff.contract`, `resid2` (level 2)
  5. `final_norm` (level 2)
  6. `lm_head` (level 2)
  7. `next_token` (`NEXT_TOKEN`, level 1)
- Per-head attention detail is stored inside `ATTENTION_PATTERN`. The `DEEP` level only changes how scenes display it. Generic-adapter steps are level 2.
- **Generation semantics must match `previous_chapters.generate()`:** crop to the context, top-k filter, temperature, multinomial or argmax, and stop at EOS. Use `torch.Generator` seeded from the profile or CLI for reproducibility.
- **Safety:** wrap everything in `torch.inference_mode()`, and restore model state afterwards.
- **Progress:** log per-pass progress with `logging`.

### 6.2 Render (Module B)

#### 6.2.1 Platform and window (`platform.py`, `window.py`)

- **Windows DPI:** before `pygame.display.init()`, call `ctypes.windll.shcore.SetProcessDpiAwareness(2)`, with a fallback to `user32.SetProcessDPIAware()`. Wrap it in `try` and make it a no-op on other OSes. Without this, the screenshare is blurry on scaled displays.
- **Window modes** (setting `window.mode`):
  - `borderless` (**default for presenting**): a `NOFRAME` window sized to `pygame.display.get_desktop_sizes()[display]`. This is screenshare-friendly. Exclusive fullscreen can appear black in some screenshare tools and makes Alt-Tab risky.
  - `windowed`: resizable, default 1600×900.
  - `exclusive`: true `FULLSCREEN`.
- `F11` toggles the window mode at runtime. The app must handle `VIDEORESIZE`, `WINDOWSIZECHANGED` and `WINDOWFOCUSLOST` without losing state.
- Set the window title and an icon.

#### 6.2.2 App loop (`app.py`)

```python
class App:
    def __init__(self, trace, presenter_settings, *, headless=False): ...
    def tick(self, dt: float, events: Sequence[pygame.event.Event]) -> None:
        # translate → commands → controller → scene.update(dt) → draw
    def run(self) -> int:  # real loop: clock.tick(fps), feeds tick(); returns exit code
```

- `tick()` is **pure with respect to time**. Tests drive it with synthetic events and `dt` values, without a real clock.
- **Idle optimization:** when no animation is running and there is no input, use `pygame.event.wait(timeout=250)` and skip redraws. This keeps CPU low during screenshare.
- Cap the frame rate at 60 FPS.

#### 6.2.3 Playlist and player (`playlist.py`, `player.py`)

`PresenterSettings` fields:

| Field | Default | Meaning |
|---|---|---|
| `detail_level` | `STANDARD` | Steps above this level are hidden |
| `detail_blocks` | `"first"` | Which blocks are expanded: `"first"`, `"first_last"`, `"all"`, `"none"`, or a list of block numbers |
| `later_pass_level` | `STORY` | Detail level used for passes after the first |
| `include_title`, `include_end` | `true` | Add the title and end cards |
| `animation_speed` | `1.0` | Animation speed multiplier |
| `advance_skips_animation` | `true` | Whether a press during an animation completes it first |

`build_playlist(trace, settings) -> list[StepRef]` decides which steps are included:
- **Expanded blocks:** include the steps with `level ≤ detail_level`, but exclude `block.summary`.
- **Collapsed blocks:** include only `block.summary`.
- **`STORY` level:** every block is collapsed.
- **Title and end cards:** added as synthetic refs.

**`StepPlayer`:**
- **Commands:** `next()`, `prev()`, `next_stage()`, `prev_stage()`, `first()`, `last()`, `goto(ref)`, `goto_block(pass, block)`, `next_pass()`.
- **Bounds:** all are bounded with no wrap-around and return whether the position changed.
- **Changing settings:** the playlist is rebuilt and the player re-anchors to the nearest step (same pass and block, closest level).

**Animation semantics:**
- `next` while the current scene is animating → `scene.finish()` first, if `advance_skips_animation` is set.
- `prev` always shows the previous step in its **end state**, with no reverse animation.

#### 6.2.4 Input (`commands.py`, `input_map.py`)

Default bindings (overridable in `presenter/*.toml` under `[keys]`):

| Command | Keys / mouse |
|---|---|
| NEXT | `→`, `Space`, `Enter`, `PageDown`, left-click on stage, on-screen ▶ |
| PREV | `←`, `Backspace`, `PageUp`, right-click on stage, on-screen ◀ |
| NEXT_STAGE / PREV_STAGE | `Shift+→` / `Shift+←` |
| FIRST / LAST | `Home` / `End` |
| NEXT_PASS | `Ctrl+→` |
| JUMP | click on a minimap segment |
| CYCLE_DETAIL | `D` |
| TOGGLE_NERD | `N` |
| CYCLE_HEAD / AVG_HEADS | `]` / `[` cycle, `A` average (attention scenes) |
| TOGGLE_HELP | `H`, `F1` |
| CYCLE_THEME | `T` |
| SPEED_UP / DOWN | `+` / `-` |
| TOGGLE_AUTOPLAY | `P` |
| TOGGLE_FULLSCREEN | `F11` |
| ESCAPE | `Esc`: close the overlay, or quit after a second press within 2 s |

Mouse handling:
- **Hover** shows tooltips.
- **Clicks on widgets** go to the widget. Only unhandled clicks become NEXT or PREV.

#### 6.2.5 Layout (`layout.py`)

- **Logical design size** is 1920×1080. `UiScale.s = min(w/1920, h/1080)`. All sizes in code are logical px multiplied by `s`.
- **Regions:**
  - `header`: about 9%, holding the minimap and HUD
  - `stage`: about 70%, holding the scene
  - `caption`: about 15%, holding the caption box and legend
  - `footer`: about 6%, holding the buttons and hint text
- **Letterboxing:** used for extreme aspect ratios.
- **Minimum sizes** for screenshare legibility:
  - body text ≥ 26 logical px
  - titles ≥ 40
  - tile ≥ 8 physical px, with a 1 px gap
  - If a grid cannot fit at its minimum size, the scene asks the `TileGrid` to **downsample further for display**, and shows `"showing N of M"`.

#### 6.2.6 Colors (`colormap.py`, `normalization.py`)

- A `Colormap` is a `(256, 3) uint8` LUT. `map(values_normalized_-1..1) -> (H, W, 3)` is fully vectorized.
- **Built-in colormaps:**
  - `diverging_blue_orange` (default for signed values): blue = negative, near-white or near-panel = 0, orange = positive. It is colorblind-safe.
  - `sequential_purple` (for probabilities and attention weights, which are in `[0, 1]`).
  - `diverging_teal_magenta` (alternative).
- **Normalizers:**
  - `per_tensor`: symmetric, using `stats.abs_p99`. This is **robust to GPT-2's huge outlier dimensions**.
  - `shared_step` (default): input, output and update tensors of a step share one scale, so changes are comparable. Weights use their own scale.
  - `global_pass`: the residual stream uses one scale per pass.
- The legend always shows the scale in use.

#### 6.2.7 Tiles (`tiles.py`)

- `TileGrid(data, axes, colormap, scale, cell_px, gap_px)` renders once into a cached `pygame.Surface`:
  - When gaps are visible (cell ≥ 6 px), fill rects per cell.
  - For very large grids, build an RGB array via LUT, then `pygame.surfarray.make_surface`, then `pygame.transform.scale` (nearest-neighbor).
- **Hit testing:** `cell_at(pos) -> (row, col) | None` maps through `axes[i].indices` to the **original** indices for tooltips. Example tooltip: `token "·cat" · dim 431 · −0.82`.
- **Highlight API** for animations: rows, columns or cells with an alpha overlay, plus a `reveal_mask` (which cells are visible yet).

#### 6.2.8 Scenes (`scenes/`)

```python
class Scene(Protocol):
    def enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None: ...
    def update(self, dt: float) -> None: ...
    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None: ...
    def finish(self) -> None: ...            # jump to end state
    @property
    def is_animating(self) -> bool: ...
    def hit_test(self, pos) -> Tooltip | None: ...
    def handle_command(self, cmd: Command) -> bool: ...   # e.g. head cycling; True if consumed
```

- `SceneContext` holds: theme, colormaps, normalizer, font manager, UiScale, the trace, the current pass, nerd mode and animation speed.
- **Scenes must be stateless across steps.** A new instance is created on every `enter`, which avoids stale-state bugs.
- The registry maps `StepKind` to a Scene class. Unknown kinds use `GenericScene`, which shows every tensor in the step as labelled tile grids.

Scene behavior (one press = the whole animation, about 0.8–1.5 s × speed; it can always be skipped):

| Kind | What the audience sees |
|---|---|
| `PROMPT` / title | The prompt types itself out, with the model name and stats ("3 layers, 4 heads, 12 numbers per word") |
| `TOKENIZE` | Text splits into colored chips, with the token ID under each chip |
| `EMBEDDING` | Chips drop into an embedding "shelf". Each pulls out a row of tiles. Position tiles slide in and add up. |
| `LAYERNORM` | A row of tiles with a wide value spread is "squeezed": a before/after strip plus a tiny spread indicator. Scale/shift weights are shown as two thin tile rows. |
| `LINEAR` | **Core scene.** The input grid (tokens × in) is on the left, the weight grid (in × out) is in the middle, and the output (tokens × out) is on the right. A highlight sweeps across the weight columns while the output columns fill in. Labels: "ff.expand → makes the list 4× longer", "ff.contract → back to normal size". |
| `QKV_PROJECTION` | One input feeds three weight grids, producing Q, K and V. Captions: question, label, content. |
| `ATTENTION_PATTERN` | A token × token heatmap, with arcs from the newest token to earlier tokens (thickness = weight). The causal mask is shown as greyed cells. `]` / `[` / `A` select heads. |
| `ATTENTION_MIX` | Attention weights blend the V rows into new rows |
| `ACTIVATION` | Tiles before and after. Negative tiles fade toward zero. A small GELU curve icon. |
| `RESIDUAL_ADD` | Two grids (shortcut + update) slide together into the output, with a "highway" arrow |
| `BLOCK_SUMMARY` | Input grid → output grid, plus a "what changed" grid. The minimap highlights the block. |
| `LOGITS` | The last token's vector meets the candidate-word rows and produces scores |
| `NEXT_TOKEN` | A bar chart of the top-k words with probabilities. The chosen word flies to the end of the text, with an optional confetti burst (theme flag). |

#### 6.2.9 Widgets and themes

- **Widgets:**
  - The **minimap** (`Input ▸ B1 ▸ B2 … ▸ Output`) collapses to `B1 … B24` with an ellipsis for deep models. Segments are clickable.
  - The **HUD** shows "Word 2 of 3 · Step 14 / 52".
  - The **caption box** shows the title plus 1–2 sentences.
  - The **nerd panel** (shown with `N`) shows the module path, the full shapes, the stats, fidelity and the reduction note.
  - Also: the legend, the ◀ ▶ buttons, tooltips, and the help overlay listing the active bindings.
- **Themes:** a `Theme` dataclass holds the colors (background, panel, text, accent, masked cell), font family and sizes, tile gap and corner radius, colormap names, confetti on/off, and an animation speed factor.
  - Built-ins: `dark` (default) and `light` (for bright projector rooms). Both meet text contrast ≥ 4.5:1.
- **Fonts:** bundle an OFL font (Noto Sans Regular and Bold) under `assets/fonts/` with `OFL.txt`. **Never rely on system fonts.** `FontManager` loads fonts via `importlib.resources` and caches them by `(family, px)`.

### 6.3 Narration

- `CaptionProvider(locale="en").caption(step, trace, pass_) -> Caption(title, body, nerd)`.
- **Lookup:** texts come from `locales/en.toml` and are looked up by `semantic`, then `kind`, then `generic`.
- **Placeholders:**
  - `{block_num}` (1-based)
  - `{n_blocks}`
  - `{n_heads}`
  - `{emb_dim}`
  - `{ff_dim}`
  - `{token}` (the newest token)
  - `{top_word}`
  - `{top_prob:.0%}`
- **Missing placeholders** never crash: render them as `?` and log a warning.
- **Tone:** short, friendly, metaphor-based ("fingerprint", "spotlight", "highway"), with no jargon in `title` / `body`. Jargon belongs in `nerd` only.
- **New languages:** a new language is only a new TOML file. It may need a font with the right glyphs, which the theme setting `font_family` covers.

### 6.4 CLI (`cli.py`)

```
tilevis record  --profile PROFILE.toml --prompt TEXT [--new-tokens 3] [--strategy greedy|sample]
                [--temperature T] [--top-k K] [--seed N] [--n-heads H] [--device auto|cpu|cuda|mps]
                [--max-features 64] [--max-tokens 24] -o OUT.trace.npz
tilevis play    TRACE.trace.npz [--presenter presenter/default.toml] [--window borderless|windowed|exclusive]
                [--theme dark|light] [--detail 1|2|3]
tilevis inspect --profile PROFILE.toml | --checkpoint X.pth     # prints inferred config, adapter, fidelity on a test prompt
tilevis check   [TRACE ...]                                     # preflight: pygame, fonts, display size, traces load & validate
tilevis demo    [--prompt TEXT]                                 # tiny random model → record in memory → play (needs [record])
```

- Exit codes: `0` success, `1` user error (with a friendly message, no traceback unless `--verbose`), `2` internal error.
- `play` and `check` must work **without torch installed**.

### 6.5 Presenter settings file (`presenter/default.toml`)

```toml
[window]
mode = "borderless"      # borderless | windowed | exclusive
display = 0
fps = 60

[view]
theme = "dark"
detail_level = 2
detail_blocks = "first"  # first | first_last | all | none | [1, 4]
later_pass_level = 1
nerd_mode = false
animation_speed = 1.0
normalizer = "shared_step"
locale = "en"

[keys]                   # optional overrides, pygame key names
next = ["right", "space", "return", "page down"]
```

---

## 7. Coding standards

- **Formatting and linting:** `ruff format` and `ruff check`.
  - Rule sets: `E, F, W, I, B, UP, SIM, RUF, PTH, N`.
  - Line length is 100.
- **Typing:**
  - Full type hints everywhere.
  - `mypy --strict` for `tilevis.core`, `tilevis.narration` and `tilevis.render`.
  - `mypy` (non-strict, `ignore_missing_imports` for torch internals) for `tilevis.probe`.
- **Docstrings:** Google style on every public class and function. Module docstrings state the module's single responsibility and its allowed dependencies.
- **No module-level side effects** except registry registration. In particular, no pygame init and no torch device queries at import time.
- **Errors:** raise subclasses of `TilevisError` with actionable messages ("what happened → why → how to fix").
- **Determinism:** seeded randomness, stable ordering, and no reliance on dict ordering from external sources.
- **Constants:** defaults live in the settings dataclasses, not scattered as magic numbers.
- **Comments:** explain *why*, not *what*. Do not remove existing comments that are unrelated to your change.
- **Commits:** one logical change per commit, with a message like `phase-N: <summary>`.

---

## 8. Testing strategy (applies to every phase)

### 8.1 Layers

| Layer | Location | Needs torch | Needs display |
|---|---|---|---|
| Unit (core, narration) | `tests/core`, `tests/narration` | no | no |
| Unit (probe) | `tests/probe` | yes | no |
| Unit (render) | `tests/render` | no | headless (`SDL_VIDEODRIVER=dummy`) |
| Integration (record → save → load → play headless) | `tests/integration` | yes | headless |
| Boundary / architecture | `tests/test_import_boundaries.py` | no | no |
| Manual | checklists in §9 | (varies) | real screen |

### 8.2 Fixtures and markers

- `tests/conftest.py`:
  - Sets `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy` **before** pygame is imported.
  - Provides a `tmp_path`-based trace directory.
  - Skips torch tests cleanly with `pytest.importorskip("torch")` when the record extra is missing.
- `tests/fixtures/fake_traces.py`: builds valid synthetic Traces of any shape (n_layers, emb_dim, T, passes, kinds) **without torch**. The render tests use only these, plus real traces in integration tests.
- `tests/fixtures/tiny_models.py`:
  - Builds `previous_chapters.GPTModel` instances from `compat_matrix.py` configs.
  - Saves them as `.pth`, plain or LoRA-wrapped, in all checkpoint styles.
  - Writes matching profiles.
- **Markers:**
  - `slow`: GPT-2-small-shaped models, long runs
  - `windows`: Windows-only behavior
  - `manual`: documentation-only placeholders
  - `perf`: performance budgets
- **Default run:** `pytest -m "not slow and not manual"`. The **phase gate** runs everything except `manual`.

### 8.3 Compatibility matrix (`tests/compat_matrix.py`), reused by every phase

| Name | emb_dim | n_layers | n_heads | ctx | qkv_bias | Variant | Purpose |
|---|---|---|---|---|---|---|---|
| `tiny_ch07` | 12 | 1 | 2 | 120 | False | plain | the exact `--test_mode` config from `gpt_instruction_finetuning.py` |
| `small_bias` | 48 | 3 | 4 | 64 | True | plain | typical student model |
| `heads_8_of_64` | 64 | 2 | 8 | 64 | False | plain | n_heads detection with many divisors |
| `head_dim_1` | 8 | 1 | 8 | 32 | False | plain | degenerate head size |
| `deep_narrow` | 32 | 12 | 4 | 64 | True | plain | many blocks: playlist, minimap |
| `lora_small` | 48 | 2 | 4 | 64 | True | LoRA r=4, α=8 | LoRA merge |
| `wrapped_ckpt` | 48 | 2 | 4 | 64 | False | `{"model_state_dict":…}` + `module.` prefix | checkpoint unwrapping |
| `gpt2_small_shape` | 768 | 12 | 12 | 1024 | True | plain, `slow` | real-size performance and memory |

### 8.4 Performance budgets (`perf` marker; measured on CPU)

| Item | Budget |
|---|---|
| Record `small_bias`, 3 passes | < 5 s |
| Record `gpt2_small_shape`, 3 passes, 8-token prompt | < 60 s, peak extra RAM < 1.5 GB |
| Trace file size for `gpt2_small_shape`, 3 passes | < 50 MB |
| Load and validate that trace | < 2 s |
| Scene `enter` (cache build) at 1920×1080 | < 100 ms |
| Average frame time while animating at 1920×1080 (headless) | < 16 ms |

### 8.5 How to write the per-phase test step

Every phase's **Test step** has six parts:
1. **New unit tests** for the phase's code.
2. A **full regression** run of all previous phases' tests. The whole suite must be green.
3. **Compatibility:** the phase's functionality is exercised against every `compat_matrix` entry it touches.
4. **Weakness probes:** adversarial and edge-case tests targeting the risks listed for that phase.
5. **Static checks:** `ruff check`, `ruff format --check`, `mypy`, and the import-boundary test.
6. **Manual checks**, where a human or a real display is needed. Record their results in the changelog.

**A phase is not complete until all six pass**, or until a documented, user-approved exception is recorded in `changelog.md` under *Known issues*.

---

## 9. Development plan

Phase N produces version `0.N.0`. Phase 8 produces `1.0.0`. Post-`1.0.0` enhancement phases increment minor versions (`1.1.0`, …). Do phases **in order**.

```
P1 Scaffolding ─► P2 Core contract ─► P3 Model loading ─► P4 Probe/recorder ─► P5 Renderer base ─► P6 Scenes ─► P7 Presentation UX ─► P8 Windows deploy ─► P9 Crisp HiDPI & auto-resolution
```

Each phase lists: **Goal**, **Tasks**, **Deliverables**, **Test step** (six parts, §8.5) and **Exit criteria**.

---

### Phase 1: Project scaffolding and tooling (`0.1.0`)

**Goal:** an installable, lint-clean, testable empty package with enforced architecture boundaries.

**Tasks**
1. `pyproject.toml`:
   - setuptools build, `src/` layout, `requires-python = ">=3.11"`
   - base deps `numpy`, `pygame-ce`
   - extras `record`, `dev`, `build`
   - console script `tilevis = tilevis.cli:main`
   - ruff, mypy and pytest config (markers, `testpaths`, `timeout = 120`)
2. Package skeleton exactly as in §4.2. Use empty modules with docstrings stating responsibility and allowed deps. Write the `__version__`.
3. `core/errors.py`: `TilevisError`, `ModelLoadError`, `ConfigInferenceError`, `TraceFormatError`, `RenderError`, `FidelityWarning` (a `UserWarning`).
4. `core/registry.py`: the generic `Registry`.
5. `logging_setup.py`: console handler; on Windows, set `errors="replace"` on the stream (`sys.stdout.reconfigure`). Add a `--verbose` flag.
6. `cli.py`: argparse with the 5 subcommands stubbed (they print "not implemented yet", exit code 1). Add `--version`. Imports are lazy.
7. `tests/test_import_boundaries.py` (AST + subprocess, §4.3).
8. `.gitignore`. Optionally `.github/workflows/ci.yml` (matrix: ubuntu/windows/macos × 3.11/3.12; installs CPU torch from `https://download.pytorch.org/whl/cpu`).
9. `tests/conftest.py` with the headless SDL env, plus the empty `compat_matrix.py` structure.

**Deliverables:** `pip install -e ".[record,dev]"` works; `tilevis --version` and `python -m tilevis --help` work.

**Test step**
1. *Unit:*
   - Registry: register, get, duplicate rejection, `override=True`, and a missing key giving a helpful error.
   - Errors hierarchy.
   - The CLI parses every subcommand, and `--version` matches `__version__`.
2. *Regression:* (n/a, first phase). The full suite is green.
3. *Compatibility:*
   - Install in fresh venvs on Python 3.11 and 3.12.
   - Install **without** the `record` extra, then confirm `tilevis --help` works and `import tilevis.render` works.
   - On Windows (CI or manual), run the same.
4. *Weakness probes:*
   - The boundary test must **fail** when a deliberately bad import is added to a temp copy of a module. Test the test via a fixture that writes a fake package.
   - Confirm `pygame` and `pygame-ce` are not both installed (a `check` helper warns if so).
   - Confirm a non-ASCII working-directory path, e.g. `C:\Users\학생\tilevis`, works. Simulate it with `tmp_path / "학생"`.
5. *Static:* ruff, format check, mypy and the boundary test are clean.
6. *Manual:* none.

**Exit criteria:** all green, and the changelog entry for 0.1.0 is written.

---

### Phase 2: Core data contract and trace format (`0.2.0`)

**Goal:** a complete, validated, versioned Trace model with a torch-free fake-trace generator.

**Tasks**
1. `core/kinds.py`: all enums from §5.1, plus helper `parse_kind(str) -> StepKind | str` (it tolerates unknown values).
2. `core/trace.py`: all dataclasses from §5.2 with read-only arrays, plus convenience methods:
   - `Trace.step_by_id`
   - `PassRecord.steps_for_block(b)`
   - `TensorSnapshot.by_role` (helper on `Step`)
3. `core/serialization.py`: `save_trace`, `load_trace`, manifest encode/decode, `MIGRATIONS` dict (empty for v1), atomic write.
4. `core/validation.py`: `validate_trace` with all invariants from §5.4. It returns a list of issues, and `strict=True` raises.
5. `core/config.py`: the `PresenterSettings` and `RecordSettings` dataclasses (defaults per §6), plus `load_toml(path) -> dict` with friendly errors and an unknown-key warning helper.
6. `tests/fixtures/fake_traces.py`: `make_fake_trace(n_layers=2, emb_dim=16, n_tokens=5, n_passes=2, n_heads=2, ff_mult=4, include_kinds=ALL, seed=0)`. It produces every `StepKind` in the template order of §6.1.11.

**Deliverables:** traces can be created, saved, loaded and validated without torch or pygame.

**Test step**
1. *Unit:*
   - Dataclass immutability: assigning to a field or writing to an array raises.
   - The orientation docstring examples hold.
   - Every validation rule has one passing test and one failing test.
   - Config defaults and TOML loading; malformed TOML gives a friendly error.
2. *Regression:* the Phase 1 suite stays green.
3. *Compatibility:*
   - Fake traces shaped like **every** `compat_matrix` entry (sizes only) round-trip losslessly: arrays are bit-equal and metadata equal.
   - Round-trip a trace whose path and token labels contain Korean, emoji and `\n` / `·` characters.
4. *Weakness probes:*
   - **hypothesis** property: random valid traces → save → load → equal.
   - Corrupt files: truncated zip, missing `__manifest__`, manifest referencing a missing array, wrong dtype, NaN in data. Each must raise `TraceFormatError`, never a raw `KeyError` or `ValueError`.
   - A pickled object array inside the npz must be rejected (`allow_pickle=False`).
   - A newer `schema_version` gives a clear error.
   - An unknown `kind` string loads fine.
   - Atomic write: simulate an exception mid-write, and confirm the original file is intact.
   - Large trace (`gpt2_small_shape`-sized fake): file size and load time stay within the §8.4 budgets.
5. *Static:* clean, and the boundary test confirms `core` imports only numpy and stdlib.
6. *Manual:* none.

**Exit criteria:** all green; the trace schema v1 is documented in `technical_document.md`; changelog updated.

---

### Phase 3: Model loading, config inference, text and adapters (`0.3.0`)

**Goal:** any teammate's model loads from a profile or checkpoint into a ready `nn.Module` plus a `ModelSpec`, with no hard-coded sizes.

**Tasks**
1. `loading/code_import.py` (§6.1.1).
2. `loading/profile.py` (§6.1.2), plus `profiles/tiny_random.toml` and `profiles/example_team_model.toml` (heavily commented).
3. `loading/checkpoint.py` (§6.1.3).
4. `loading/config_infer.py` (§6.1.4).
5. `loading/lora.py` (§6.1.5).
6. `loading/factory.py` (§6.1.6).
7. `text/tokenizer.py`, `text/templates.py` (§6.1.7).
8. `adapters/base.py`, `gpt_scratch.py`, `generic.py`, and auto-selection (§6.1.8).
9. `weights.py`: `WEIGHT_EXTRACTORS` for `nn.Linear` (transposed to in×out), LoRA wrapper (effective weight), `nn.Embedding`, LayerNorm-like (`scale` / `shift`, or `weight` / `bias`).
10. `tilevis inspect` implemented. It prints:
    - profile, resolved paths, inferred cfg (marking which values were inferred and which were explicit)
    - adapter and confidence
    - parameter count
    - block roles table
    - warnings
11. `tests/fixtures/tiny_models.py`: build and save every `compat_matrix` variant, in all checkpoint styles.

**Deliverables:** `tilevis inspect --profile profiles/tiny_random.toml` and `tilevis inspect --checkpoint some.pth` both work.

**Test step**
1. *Unit:*
   - Every inference rule.
   - Prefix stripping and container unwrapping.
   - The LoRA merge math: the merged model's logits equal the wrapped model's logits within `1e-5`.
   - Template output equals the original `format_input` text. Import `gpt_instruction_finetuning.format_input` by path in the test, for comparison only.
   - Tokenizer display rules: space, newline, partial UTF-8.
   - Adapter role resolution, including `[adapter.roles]` overrides.
2. *Regression:* Phases 1–2 green. Fake traces still validate.
3. *Compatibility:* for **every** `compat_matrix` entry, save a checkpoint, then load it via profile **and** via `--checkpoint`, and check:
   - (a) the inferred cfg equals the true cfg (except `n_heads`, when it is not given),
   - (b) `load_state_dict(strict=True)` succeeds,
   - (c) the logits equal those of the original in-memory model **bit-for-bit** on CPU,
   - (d) `gpt_scratch` is selected with confidence ≥ 0.9,
   - (e) `generic` also produces a sensible leaf list when forced.
4. *Weakness probes:*
   - Profile contradicts the checkpoint (`n_layers`, `emb_dim`) → `ConfigInferenceError` with both values in the message.
   - LoRA checkpoint without `[model.lora]` → loads, and logs a warning.
   - Wrong vocab (a custom vocab of 1000 with GPT-2 tokenizer ids) → clear `ModelLoadError` at encode time.
   - A missing checkpoint path, or a path with spaces and Korean characters.
   - A pickled full module with `allow_pickle=false` → clear error suggesting the flag.
   - Two different `previous_chapters.py` copies imported in the same process don't clash.
   - `sys.path` is unchanged after import, even when the user module raises.
   - A model without `pos_emb` (a custom subclass in a test fixture) → the adapter sets the position embedding to `None`.
5. *Static:* clean. The boundary test confirms `probe` has no pygame and `core` has no torch.
6. *Manual:* run `tilevis inspect` on at least one **real** teammate checkpoint if available, and record the output summary in the changelog.

**Exit criteria:** all green; changelog updated with a list of verified checkpoint styles.

---

### Phase 4: Probe: hooks, builders, reduction, generation, recorder (`0.4.0`)

**Goal:** `tilevis record` produces a valid, faithful Trace for any compatible model, covering all internal layers **and** the next-word prediction, for each pass.

**Tasks**
1. `hooks.py`, `capture.py` (§6.1.9).
2. `reduce.py` (§6.1.10).
3. All builders in `builders/` (§6.1.9), including attention recompute, fidelity and n_heads detection.
4. `generation.py` (§6.1.11), mirroring `previous_chapters.generate()`.
5. `recorder.py`, plus the `tilevis record` command. Also write the `ModelSpecInfo` and settings into the Trace.
6. Generate and commit `traces/samples/tiny_demo.trace.npz` (`small_bias`, random seed 123, prompt "Every effort moves you", 3 tokens) and `traces/samples/tiny_ch07.trace.npz`.

**Deliverables:** `tilevis record --profile profiles/tiny_random.toml --prompt "Every effort moves you" -o traces/demo.trace.npz`.

**Test step**
1. *Unit:*
   - **HookManager:** handles are removed after normal exit **and** after an exception. Call counters are correct for modules called twice.
   - Reducer index plans are deterministic, strictly increasing and within bounds. The same indices are used across all steps for the same axis family.
   - **Stats** are computed on the full tensor.
   - **Builders** are fed synthetic captures and produce the expected kinds, levels and roles.
2. *Regression:*
   - Phases 1–3 green.
   - Recorded traces pass `validate_trace` and round-trip through Phase 2 serialization.
   - `tilevis inspect` still works.
3. *Compatibility:* for **every** `compat_matrix` entry (the `gpt2_small_shape` entry under `slow`), check:
   - (a) **Non-interference:** logits with the recorder active are `torch.equal` to logits without it, model `training` flag and `requires_grad` flags are unchanged, and no hooks remain (`_forward_hooks` / `_forward_pre_hooks` are empty on every module).
   - (b) **Coverage:** every pass contains the full template order for **all** `n_layers` blocks, plus a `NEXT_TOKEN` step.
   - (c) **Fidelity:** attention and both residual steps are `exact`, and the LM-head step matches the real logits.
   - (d) **Auto-detection:** when `n_heads` is omitted, it is detected correctly (`heads_8_of_64`, `head_dim_1`).
   - (e) **Generation:** greedy tokens equal `previous_chapters.generate(..., temperature=0)` tokens, and seeded sampling equals `generate` with the same `torch.manual_seed` semantics.
   - (f) The LoRA model trace matches the merged model.
   - (g) `--adapter generic` produces a valid trace of generic and linear steps.
4. *Weakness probes:*
   - **1-token prompt:** n_heads stays `unverified` with a warning, and nothing crashes.
   - **Prompt longer than `context_length`:** left-cropped, noted in `meta`.
   - **EOS generated at pass 1:** the trace ends cleanly and the last `prediction` shows EOS.
   - **Attention changed:** a test fixture subclass removes the scaling, so fidelity is `failed` and a black-box `GENERIC` step is emitted, with no crash.
   - **NaN injected** into a weight: values are replaced, the flag is set, and the trace still validates.
   - **`max_features` larger than `emb_dim`:** handled with no padding.
   - **`max_tokens` smaller than T:** the last tokens are kept and labels align.
   - **Memory and time budgets** (`perf`).
   - Recording twice on the same model object gives identical traces (determinism).
   - **Device:** `--device mps` / `cuda` is skipped if unavailable, but CPU and accelerator traces agree within `1e-3` where available.
5. *Static:* clean.
6. *Manual:* record traces from at least two **real** teammate models with different sizes. Check `tilevis inspect` and the trace step counts, and note them in the changelog.

**Exit criteria:** all green; sample traces committed; changelog updated, including the measured perf numbers.

---

### Phase 5: Renderer foundation, navigation, tiles and generic scene (`0.5.0`)

**Goal:** `tilevis play` opens a DPI-aware window (or runs headless), navigates any Trace forward and backward across passes, blocks and detail levels, maps arrays to colorblind-safe tile surfaces, and renders every step cleanly through `GenericScene`, captions and a basic HUD—with zero PyTorch dependency.

**Tasks**
1. `narration/captions.py` and `narration/locales/en.toml` (§6.3), covering every `Semantic` and `StepKind` in §5.1 with audience-friendly metaphors and nerd-mode strings.
2. Bundle OFL font files (Noto Sans Regular and Bold + `OFL.txt`) under `src/tilevis/assets/fonts/`, and implement `render/text.py` (`FontManager` with `(family, size_px)` cache).
3. `render/platform.py` and `render/window.py` (§6.2.1): Windows High-DPI awareness, `borderless` / `windowed` / `exclusive` modes, resize and focus-loss handling.
4. `render/themes/base.py` and `render/themes/builtin.py` (§6.2.9): `Theme` dataclass plus `dark` and `light` themes.
5. `render/colormap.py` and `render/normalization.py` (§6.2.6): `COLORMAPS` (`diverging_blue_orange`, `sequential_purple`, `diverging_teal_magenta`) and `NORMALIZERS` (`per_tensor`, `shared_step`, `global_pass`).
6. `render/tiles.py` (§6.2.7): `TileGrid` surface caching, display-time downsampling when tiles would fall below minimum physical size, `cell_at()` hit testing back to original `AxisInfo.indices`, and highlight/reveal overlays.
7. `render/layout.py` (§6.2.5): `UiScale` (1920×1080 logical design space), regions (`header`, `stage`, `caption`, `footer`) and letterboxing.
8. `render/playlist.py` and `render/player.py` (§6.2.3): `StepRef`, `build_playlist` (`detail_level`, `detail_blocks`, `later_pass_level`, title/end cards) and `StepPlayer` navigation + re-anchoring when settings change.
9. `render/commands.py`, `render/input_map.py` (§6.2.4) and `presenter/default.toml` (§6.5).
10. `render/animation.py`: `Tween`, `Timeline`, easing functions (`linear`, `ease_out_cubic`, `ease_in_out_quad`), speed multiplier and `finish()`.
11. `render/scenes/base.py`, `render/scenes/registry.py` and `render/scenes/generic.py` (§6.2.8).
12. Initial `widgets/caption_box.py` and `widgets/hud.py` (pass counter, step counter).
13. `render/app.py` (§6.2.2) with pure `tick(dt, events)` and idle-throttled `run()`, wired to `tilevis play`.

**Deliverables:** `tilevis play traces/samples/tiny_demo.trace.npz` opens and steps cleanly forward and backward through every recorded step using colored tile grids and plain-English captions.

**Test step**
1. *Unit:*
   - **Captions:** every `Semantic` and `StepKind` resolves without missing-key warnings; missing placeholder values render as `?` without raising.
   - **Colormap & Normalizers:** **hypothesis** property tests for arbitrary finite `float32` arrays (including all-zeros, single-element, constant negative, extreme outliers `±1e6`) → output RGB shape `(H, W, 3)`, `uint8`, symmetric around zero for diverging maps.
   - **TileGrid:** `cell_at()` round-trips every visible cell center to the exact `(row, col)` and original `AxisInfo.indices`, and returns `None` in gaps and outside bounds.
   - **Playlist & StepPlayer:** **hypothesis** stateful/sequence tests applying random sequences of `next`, `prev`, `next_stage`, `prev_stage`, `first`, `last`, `next_pass`, `goto_block` and detail-level switches → cursor always valid, never wraps, re-anchors to the same pass/block when detail changes.
   - **Animation:** `Timeline` progresses monotonically with `dt`, respects `animation_speed`, and `finish()` jumps immediately to `t = 1.0`.
   - **InputMap:** default keys, mouse clicks and custom `[keys]` overrides map to expected `Command`s; unrecognized keys are ignored cleanly.
2. *Regression:* Phases 1–4 test suites pass completely; sample traces from Phase 4 load and step end-to-end inside headless `App`.
3. *Compatibility:*
   - Headless `App` steps from start to end and back to start on synthetic traces matching **every** `compat_matrix` configuration (including 12-layer `deep_narrow` and `gpt2_small_shape`) and on both committed Phase 4 sample traces.
   - Run the entire `tests/render` and `tests/narration` suite in an environment (or subprocess hook blocking `torch`) proving `tilevis play` never imports `torch`.
4. *Weakness probes:*
   - **Extreme viewport sizes:** `tick()` and `draw()` at 800×600, 1280×720, 1366×768, 1920×1080, 2560×1440, 3840×2160 and ultrawide 3440×1440—no negative-size `pygame.Rect`, no overlapping header/stage/caption regions, and tiles auto-downsample with `"showing N of M"` when space is tight.
   - **Rapid resize mid-playback:** dispatch `VIDEORESIZE` / `WINDOWSIZECHANGED` events between steps and mid-animation; state and cursor remain intact.
   - **Unknown `StepKind` and `Semantic`:** a forward-compatible trace step with `kind="future_op"` renders via `GenericScene` and fallback caption without crashing.
   - **Single-step playlist & EOS-only pass:** bounded navigation does not crash on empty blocks or `prediction=None`.
   - **Double-Esc behavior:** first `Esc` with no overlay arms quit prompt; `Esc` after > 2.0 s resets timer instead of quitting; second `Esc` within 2.0 s exits cleanly.
5. *Static:* `ruff check`, `ruff format --check`, `mypy --strict` on `core`, `narration`, `render`, plus `test_import_boundaries.py`.
6. *Manual:* launch `tilevis play traces/samples/tiny_demo.trace.npz` in windowed and borderless modes on the dev machine; verify crisp font rendering, tile legibility and smooth forward/back stepping. Record results in `changelog.md`.

**Exit criteria:** all six test parts pass; changelog updated for `0.5.0`.

---

### Phase 6: Dedicated scenes: matmul, attention, residual and prediction (`0.6.0`)

**Goal:** implement all dedicated, animated scenes in §6.2.8 so non-experts can visually follow how input vectors are transformed by each layer's weights (R1, R2, R3) and how the model picks the next word (R4, R5).

**Tasks**
1. `scenes/title.py`: title card (prompt preview, friendly model summary) and end card (full generated text highlighted by pass).
2. `scenes/tokenize.py`: `PROMPT` and `TOKENIZE` scenes (text splitting into colored token chips with token IDs).
3. `scenes/embedding.py`: `EMBEDDING` scene (token chips pulling out `token_vectors` rows, `position_vectors` sliding in, elementwise sum into input activations).
4. `scenes/layernorm.py`: `LAYERNORM` scene (before/after activation tile grids with spread indicator, plus `scale` and `shift` parameter strips).
5. `scenes/matmul.py`: `LINEAR` core scene (`input` tiles × `weight` tiles → `output` tiles; synchronized column-sweep highlight across weight columns as output columns fill in; semantic-aware framing for `block.att.out`, `block.ff.expand`, `block.ff.contract`).
6. `scenes/qkv.py`: `QKV_PROJECTION` scene (one normalized input branching through three weight grids into Question `Q`, Label `K` and Content `V` tile grids).
7. `scenes/attention.py`: `ATTENTION_PATTERN` scene (causal-masked `T × T` heatmap with greyed upper-triangle cells, spotlight arcs from the newest token to attended earlier tokens, head selector `]` / `[` and head-average toggle `A`).
8. `scenes/attention_mix.py`: `ATTENTION_MIX` scene (attention weights blending `V` rows into context vectors).
9. `scenes/activation.py`: `ACTIVATION` scene (before/after tile grids where negative tiles mute toward zero, paired with a compact GELU curve badge).
10. `scenes/residual.py`: `RESIDUAL_ADD` scene (`shortcut` "highway" stream + `update` stream sliding together into summed output tiles).
11. `scenes/block_summary.py`: `BLOCK_SUMMARY` scene (block input → block output alongside a `output − input` "what changed" tile grid for collapsed blocks).
12. `scenes/logits.py` and `scenes/prediction.py`: `LOGITS` scene (last token's vector × top-k candidate rows of `out_head`) and `NEXT_TOKEN` scene (horizontal probability bar chart of top-k words, chosen word animating onto the running text, optional theme-controlled confetti).
13. Register every scene in `SCENES` (`scenes/registry.py`) and refine `locales/en.toml` captions to match visual metaphors.

**Deliverables:** every `StepKind` in §5.1 renders with its dedicated scene and one-keypress whole-layer animation.

**Test step**
1. *Unit:*
   - Parameterized scene contract test over **every** registered `StepKind`:
     - `enter(..., animate=True)` starts with `is_animating == True` (or cleanly `False` if `animation_speed` is instant),
     - `update(dt)` in 16 ms increments terminates (`is_animating == False`) within 5 s of simulated time,
     - `finish()` immediately sets `is_animating == False` and produces the exact same rendered surface as stepping `update()` to completion,
     - `enter(..., animate=False)` (used by `PREV`) starts directly in the end state,
     - `hit_test(pos)` returns valid `Tooltip` objects over tile grids and `None` on empty background.
   - `AttentionScene` head commands: `CYCLE_HEAD` (`]` / `[`) wraps cleanly across `0 .. n_heads-1`, `AVG_HEADS` (`A`) switches to mean view, and unhandled commands return `False`.
   - `MatmulScene` orientation check: verify displayed `(T, in)` × `(in, out)` → `(T, out)` dimensions and labels align with `Step` snapshots.
   - `PredictionScene`: bar lengths are monotonic with `candidate_probs`, and the `chosen_id` (even when sampling picks a non-argmax candidate) is the highlighted winner.
2. *Regression:* Phases 1–5 suites stay green; `StepPlayer` animation-skip semantics (`NEXT` while animating calls `finish()`, second `NEXT` advances; `PREV` shows end state immediately) verified across all new scenes.
3. *Compatibility:*
   - Step headlessly through complete playlists (`STORY`, `STANDARD`, `DEEP`) for traces generated from **every** `compat_matrix` model (`tiny_ch07`, `small_bias`, `heads_8_of_64`, `head_dim_1`, `deep_narrow`, `lora_small`, `wrapped_ckpt`, `gpt2_small_shape`), rendering at least 3 animation frames (`t=0`, mid, end) per step without error.
4. *Weakness probes:*
   - **Edge shapes:** 1-token prompt (`T=1`, attention arcs degenerate to self-loop), max-cropped prompt (`T=24` with long/Korean/newline tokens), `head_dim=1`, `n_heads=1`, missing `position_vectors` (`pos_emb=None`), `bias=None` vs `bias` present in `LINEAR`/`QKV_PROJECTION`, and `prediction=None` on EOS stop.
   - **Rapid input mashing:** 200 alternating `NEXT` / `PREV` / `CYCLE_HEAD` / `CYCLE_DETAIL` commands with random `dt` values (`0.0`, `0.001`, `0.5`)—no stale state across steps and no exceptions.
   - **Performance budgets (§8.4, `perf`):** at 1920×1080 headless, every scene's `enter()` completes in `< 100 ms` and average `tick()` + `draw()` frame time during animation is `< 16 ms` even on `gpt2_small_shape` reduced snapshots.
5. *Static:* ruff, format check, `mypy --strict` and import-boundary tests pass.
6. *Manual:* visually inspect every scene type on `tiny_demo.trace.npz` and a multi-layer model trace; verify animations read intuitively at 1× speed and remain legible over a test screenshare window. Log notes in `changelog.md`.

**Exit criteria:** all six test parts pass; changelog updated for `0.6.0`.

---

### Phase 7: Presentation UX, widgets, nerd mode and polish (`0.7.0`)

**Goal:** complete all interactive HUD widgets, mouse-only controls, tooltips, nerd mode, themes and `tilevis demo` so presenting on a single screen with keyboard or mouse is effortless (R5, R10).

**Tasks**
1. `widgets/minimap.py`: top pipeline bar (`Input ▸ B1 ▸ B2 … ▸ Output`) that collapses middle blocks with an ellipsis when `n_layers` exceeds horizontal budget (e.g., 12–48 layers), highlights the active stage/block, and emits `JUMP` on click.
2. `widgets/legend.py`: active colormap gradient bar with formatted numeric endpoints (`−scale`, `0`, `+scale` or `0%`–`100%`).
3. `widgets/buttons.py`: on-screen clickable controls (`◀ Prev`, `Next ▶`, detail-level pill, nerd-mode badge, `?` help button) with hover states so the entire presentation can be driven by mouse alone if desired.
4. `widgets/tooltip.py`: hover card showing original unreduced coordinates (token text, original feature/head index, candidate word) and exact float value, automatically flipped/clamped near screen edges.
5. `widgets/nerd_panel.py`: toggled with `N`; displays `module_path`, `Step.id`, `full_shape` vs displayed slice, reduction strategy, `TensorStats` (`min`, `max`, `mean`, `std`, `abs_p99`) and `Fidelity` badge/error.
6. `widgets/help_overlay.py`: modal cheat sheet toggled with `H` / `F1` (or `?` button), generated dynamically from active `InputMap` bindings; `Esc` closes the overlay before arming application exit.
7. Live toggles in `App`: theme switching (`T` between `dark` and `light` with instant surface cache invalidation), autoplay (`P`) with configurable pause per step, animation speed (`+` / `-`), detail cycling (`D`).
8. Implement `tilevis demo [--prompt TEXT]` (builds `tiny_random` in memory, records 3 passes, launches `App`).

**Deliverables:** complete presenter UI with clickable minimap, on-screen buttons, edge-clamped tooltips, nerd panel, help overlay, light/dark themes and `tilevis demo`.

**Test step**
1. *Unit:*
   - **Minimap:** layout and hit-testing for `n_layers` in `{1, 3, 12, 24, 48}`; clicking any visible block segment jumps `StepPlayer` to the first playlist item of that block in the current pass.
   - **Buttons & event routing:** clicking a widget consumes the mouse event and triggers only that widget's command (never accidentally triggering stage `NEXT`/`PREV`).
   - **Tooltip positioning:** tooltips triggered at all four corners `(0, 0)`, `(W-1, 0)`, `(0, H-1)`, `(W-1, H-1)` stay 100% inside `0..W × 0..H`.
   - **Nerd panel & Legend:** accurately reflect `full_shape`, `reduction`, `Fidelity` (including warning styling on `approximate`/`unverified`/`failed`) and active normalizer scale.
   - **Theme contrast check:** automated WCAG relative-luminance test verifying primary and secondary text colors against background and panel colors achieve contrast ratio `≥ 4.5:1` in both `dark` and `light` themes.
2. *Regression:* Phases 1–6 suites remain green; keyboard navigation and scene animations unaffected when overlays are closed.
3. *Compatibility:*
   - Full headless walkthrough (clicking every button, toggling `N`, `T`, `D`, `H`, `P`, clicking minimap segments) across traces from all `compat_matrix` models, plus `tilevis demo` smoke test.
4. *Weakness probes:*
   - **Cache invalidation:** toggling `T` (theme) or `F11` / resize mid-step rebuilds `TileGrid` and font surfaces cleanly with no stale colors or memory growth across 50 toggles.
   - **Overlay precedence:** while `help_overlay` is open, stage clicks and `Esc` close the overlay rather than advancing steps or quitting the app.
   - **Mouse-only full run:** drive an entire presentation from title card to end card and back using only synthetic `MOUSEMOTION` and `MOUSEBUTTONDOWN` events (zero keyboard events).
   - **Long module paths & labels:** nerd panel and tooltips truncate or wrap gracefully when given deeply nested paths or 50-character token strings.
5. *Static:* ruff, format check, `mypy --strict` and import-boundary tests pass.
6. *Manual:* perform a timed 5-minute mock presentation run switching between Story/Standard detail, Nerd Mode and Light/Dark themes. Record UX observations in `changelog.md`.

**Exit criteria:** all six test parts pass; changelog updated for `0.7.0`.

---

### Phase 8: Windows lecture-hall readiness, preflight, packaging and docs (`1.0.0`)

**Goal:** guarantee zero-surprise execution on the Windows lecture-hall PC (screenshare over one monitor, keyboard + mouse), ship preflight diagnostics and launchers, finalize contributor/user documentation, and release `1.0.0` (R10, R11).

**Tasks**
1. Implement `tilevis check [TRACE ...]`:
   - Reports Python version, OS, SDL version, `pygame-ce` status (and warns explicitly if `pygame` conflicts), bundled font integrity, detected desktop resolutions and DPI scaling status.
   - Loads and validates every supplied `.trace.npz` (defaulting to `traces/samples/*.trace.npz`), reporting model summary, pass/step counts, fidelity warnings and load time.
   - Returns exit code `0` only if every check passes.
2. Create `scripts/play_demo.bat` (UTF-8 `chcp 65001`, activates local venv or standalone player if present, runs `tilevis check` quick-verify then `tilevis play`, `pause`s on non-zero exit so errors never vanish in a flashing console window) and `scripts/play_demo.sh`.
3. Add optional PyInstaller player build configuration (`packaging/tilevis_player.spec` + instructions) that bundles **only** playback modules (`core`, `narration`, `render`, `assets`, `presenter`, sample traces) and explicitly excludes `torch`, `tiktoken`, `matplotlib`, `pytest`, `hypothesis`, `mypy` so the lecture-hall folder stays small and dependency-free.
4. Write `README.md`:
   - 60-second quick start for playing existing traces on the lecture-hall PC,
   - step-by-step guide for teammates to create a profile and record their `.pth` model,
   - presenter keyboard & mouse controls cheat sheet,
   - troubleshooting table (blurry DPI, screenshare black window → use `borderless`, Korean path encoding).
5. Finalize and cross-check `technical_document.md` and `changelog.md` against the finished `1.0.0` codebase (ensuring zero drift between docs and code, and verifying `git status` shows `ch07/` completely untouched).

**Deliverables:** `1.0.0` release with `tilevis check`, launcher scripts, optional Windows standalone spec, `README.md`, up-to-date `technical_document.md` and complete `changelog.md`.

**Test step**
1. *Unit:*
   - `tilevis check` returns `0` on valid environment + sample traces, and returns `1` with actionable messages on a corrupt trace, missing trace path, missing font asset (simulated via monkeypatch) or simulated `pygame` package conflict.
   - Verify PyInstaller spec exclusion list and CLI entry-point wiring.
2. *Regression:* run the **entire** test suite across Phases 1–8, including `slow` and `perf` markers (`pytest -m "not manual"`), with coverage report (`≥ 85%` overall, `≥ 90%` on `core` and `narration`).
3. *Compatibility:*
   - Clean-room playback test: create a fresh virtual environment with `pip install .` (base runtime deps **only**—no `record`, no `dev`), copy sample traces into a directory with spaces and Korean characters (e.g., `발표 자료/traces/`), and run `tilevis check` + headless `tilevis play`.
   - Verify `git diff ch07/` and `git status ch07/` are completely empty (R6).
4. *Weakness probes:*
   - **Read-only directory playback:** chmod trace directory read-only (simulating locked USB drive or lecture-hall network share) and verify `tilevis check` and `tilevis play` never attempt writes alongside the trace.
   - **Focus loss & mode toggling:** simulate `WINDOWFOCUSLOST` / `WINDOWFOCUSGAINED` (`Alt+Tab` during screenshare) and repeated `F11` mode switches (`borderless` ↔ `windowed`) without crashes, stuck modifier keys (`Shift`/`Ctrl`), or lost cursor state.
   - **CLI user-error hygiene:** pass invalid flags, missing files, bad TOML, non-trace `.zip` files and unsupported schema versions to every CLI subcommand—confirm exit code `1` and clean single-paragraph error messages with **no** raw Python tracebacks unless `--verbose` is set.
5. *Static:* `ruff check`, `ruff format --check`, `mypy` across all packages, and `test_import_boundaries.py` all pass with zero warnings.
6. *Manual (Windows lecture-hall rehearsal checklist):*
   - Run `scripts\play_demo.bat` on Windows at 125% or 150% display scaling.
   - Start a real screenshare (Zoom / Teams / Discord / browser) of the screen, verify `borderless` mode is sharp, responsive to mouse & keyboard, and handles `Alt+Tab` cleanly.
   - Play through at least two teammate model traces from prompt to final next-word prediction.
   - Record rehearsal confirmation in `changelog.md`.

**Exit criteria:** all six test parts pass; `ch07/` untouched; `changelog.md` finalized for `1.0.0`.

---

### Phase 9: Native High-DPI rendering, automatic resolution detection, and crisp 1080p viewport (`1.1.0`)

**Goal:** eliminate viewport haziness/blurriness across standard 1080p (`1920×1080`), Windows HiDPI (`125%` / `150%` / `200%`), and macOS Retina displays by rendering directly at the display's physical backing-pixel resolution, automatically detecting the active monitor resolution, and keeping all text, borders, and tile grids pixel-sharp (R1, R5, R9, R10, R11).

**Tasks**
1. `render/platform.py` — Early DPI awareness, SDL hints & automatic display resolution detection:
   - Update `enable_high_dpi()` to set `os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")` and `os.environ.setdefault("SDL_VIDEO_HIGHDPI_DISABLED", "0")` **before** any SDL video or font subsystem initializes.
   - On Windows, attempt `user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))` (`DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2`) first, falling back to `shcore.SetProcessDpiAwareness(2)` and then `user32.SetProcessDPIAware()`.
   - Add `DisplayInfo` dataclass and `detect_display_info(display: int = 0) -> DisplayInfo` returning the detected desktop logical dimensions `(desktop_w, desktop_h)` and a smart default windowed resolution (`default_windowed_size(desktop_w, desktop_h)`): exact `(1920, 1080)` when the desktop exceeds 1080p, or a crisp 16:9 fit (~85% of desktop bounds, minimum `800×450`) when running in `windowed` mode on a `1920×1080` or smaller display, replacing the hardcoded `(1600, 900)` fractional default.
2. `render/window.py` — Native High-DPI `pygame.Window` backing surface & coordinate mapping:
   - Upgrade `WindowManager` to create and manage a `pygame-ce` `pygame.Window` with `allow_high_dpi=True` (using `window.get_surface()` and `window.flip()`), while preserving a clean fallback to `pygame.display.set_mode()` when `pygame.Window` is mocked or unavailable in legacy driver environments.
   - Separate and track both `logical_size: tuple[int, int]` (window size in OS points) and `drawable_size: tuple[int, int]` (`surface.get_size()` in physical backing pixels, which `self.size` reports for `UiScale`), plus `dpi_scale: tuple[float, float] = (drawable_w / logical_w, drawable_h / logical_h)`.
   - Add `WindowManager.map_point(pos: tuple[int, int]) -> tuple[int, int]` to convert mouse event coordinates (`MOUSEMOTION.pos`, `MOUSEBUTTONDOWN.pos`) from logical window space into physical drawable surface pixels (`1:1` identity on standard 1080p `1.0×` displays; `2.0×` on Retina displays).
   - Add `WindowManager.present() -> None` calling `self._window.flip()` (or `pygame.display.flip()` on fallback) so `App.run()` flips the active High-DPI surface cleanly.
3. `render/layout.py` — Exact `1.0` scale snapping on 1080p viewports:
   - In `UiScale.from_viewport(viewport_w, viewport_h)`, snap `s` to exact `1.0` when `abs(s - 1.0) < 0.01` (e.g., on `1920×1080` or `1920×1200` displays) so `px()` and `font_px()` use exact unscaled integer pixel sizes with zero rounding drift.
4. `render/app.py` — Early DPI init, mouse coordinate scaling & resize scene-context refresh:
   - Call `enable_high_dpi()` at the very beginning of `App.__init__()` *before* `pygame.font.init()`.
   - When `viewport_size` is `None`, allow `WindowManager` to auto-select its resolution from `detect_display_info(display)` rather than passing a hardcoded `(1600, 900)`.
   - Route `MOUSEMOTION` and `MOUSEBUTTONDOWN` positions through `self.window.map_point(pos)` before hit-testing HUD buttons, minimap segments, and scene tooltips.
   - In `App.resize()` and `Command.TOGGLE_FULLSCREEN`, refresh `self.scene`'s `SceneContext` and invalidate `_cached_stage_size` (via `_enter_current_step(animate=False)` while preserving timeline progress) so all `TileGrid`s and fonts immediately re-render at the new physical pixel resolution.
5. `render/scenes/common.py` — Crisp stroke and panel geometry:
   - In `draw_bezier_arc()`, remove the redundant `pygame.draw.aalines()` pass drawn over the center of thick (`thickness > 1`) polylines, which previously blended a fuzzy seam down the middle of attention arcs.
6. `check.py` — Preflight High-DPI & drawable resolution diagnostics:
   - Update `check_display_modes()` to report the detected desktop size alongside High-DPI backing resolution support.

**Deliverables:** `1.1.0` release with native High-DPI `pygame.Window` rendering, automatic monitor resolution detection, `1:1` crisp 1080p viewport edges/text, accurate High-DPI mouse coordinate mapping, and updated `changelog.md`.

**Test step**
1. *Unit:*
   - `platform.py`: verify `enable_high_dpi()` sets SDL DPI environment hints and tries `SetProcessDpiAwarenessContext` → `SetProcessDpiAwareness` → `SetProcessDPIAware`; verify `detect_display_info()` and `default_windowed_size()` across `1366×768`, `1512×982`, `1920×1080`, `2560×1440`, and `3840×2160` simulated monitors.
   - `window.py`: verify `WindowManager` tracks `logical_size`, `drawable_size`, and `dpi_scale`, and that `map_point()` maps `(x, y)` accurately for both `1.0×` (`1920×1080` surface on `1920×1080` window) and `2.0×` (`3024×1964` surface on `1512×982` window) DPI scales.
   - `layout.py`: verify `UiScale.from_viewport(1920, 1080)` produces exact `s == 1.0`, `canvas_rect == Rect(0, 0, 1920, 1080)`, and integer-aligned `px()` / `font_px()`.
2. *Regression:* full test suite across Phases 1–9 (`pytest -m "not manual"`) passes with coverage `≥ 85%` overall (`≥ 90%` on `core` and `narration`), including all 6 committed sample traces (`tiny_ch07`, `tiny_demo`, `big_en`, `big_es`, `big_ja`, `big_ko`).
3. *Compatibility:*
   - Render all 8 `compat_matrix` models and all 6 sample traces (including `big_ja` and `big_ko` CJK traces) headlessly at `(1920, 1080)`, `(1512, 982)`, `(2560, 1440)`, and `(3024, 1964)` drawable resolutions with zero errors or clipped text.
   - Verify `git diff ch07/` and `git status ch07/` remain empty (R6).
4. *Weakness probes:*
   - **Simulated 2× Retina / 150% Windows HiDPI click accuracy:** simulate a window where `logical_size = (1280, 720)` and `drawable_size = (1920, 1080)` (`dpi_scale = (1.5, 1.5)`) or `(1512, 982)` → `(3024, 1964)` (`dpi_scale = (2.0, 2.0)`); verify synthetic `MOUSEBUTTONDOWN` events in logical coordinates hit the exact on-screen footer buttons and minimap segments drawn in physical coordinates.
   - **Live resize & `F11` mode switch crispness:** resize `App` from `1280×720` → `1920×1080` → `3024×1964` mid-step and verify `scene._ctx.scale`, `TileGrid.cell_px`, and font sizes update immediately without stale low-res surfaces.
5. *Static:* `ruff check src tests`, `ruff format --check src tests`, `mypy --strict` on `core`/`narration`/`render`, `mypy` on `probe`/`check.py`, and `test_import_boundaries.py` pass with zero errors.
6. *Manual:* launch `tilevis play traces/samples/big_en.trace.npz` and `traces/samples/big_ja.trace.npz` in both `borderless` and `windowed` modes; verify razor-sharp text and tile edges at 1080p and on HiDPI/Retina displays, and record results in `changelog.md`.

**Exit criteria:** all six test parts pass; `ch07/` untouched; `changelog.md` updated for `1.1.0`.

---

## 10. Maintaining `changelog.md`

`changelog.md` is the project's permanent engineering log. Every agent working on any phase **must** keep it accurate and up to date.

### 10.1 When to update `changelog.md`

1. **At the end of every phase (mandatory gate):** Phase `N` is not complete until its `0.N.0` (or `1.0.0` for Phase 8) section in `changelog.md` is filled out with real results.
2. **Immediately when a design decision is made:** if an ambiguity is resolved with the user (Ground Rule 12) or an implementation trade-off is chosen, log it under `[Unreleased]` right away so context is never lost across sessions.
3. **Whenever dependencies or the Trace schema change:** Ground Rules 3 and 8 require logging any new dependency or `TRACE_SCHEMA_VERSION` bump (prefixed with `**BREAKING:**`).

### 10.2 Required format for each phase entry

Keep `[Unreleased]` at the top of `changelog.md`, followed by completed phases in reverse chronological order (newest release first, with a phase roadmap/status table at the top for quick orientation). Each completed phase entry must use this template:

```markdown
## [0.N.0] - YYYY-MM-DD — Phase N: <Phase Title>

**Requirements addressed:** R1, R6, ...

### Added
- Concrete modules, classes, CLI subcommands, profiles, or assets created.

### Changed
- Updates to existing modules, configs, or interfaces (prefix schema/API breaks with **BREAKING:**).

### Fixed
- Bugs or edge cases discovered and resolved during the phase's test step.

### Decisions & Rationale
- Key technical or UX choices made during this phase and *why* they were chosen.

### Verification & Test Summary
- **Automated tests:** `<passed>` passed (`<new>` new in this phase), coverage `<X>%`.
- **Compatibility matrix:** list of `compat_matrix` models verified.
- **Performance (if applicable):** measured timings and trace sizes vs §8.4 budgets.
- **Static checks:** `ruff`, `mypy`, `test_import_boundaries` status.
- **Manual checks:** summary of manual verification performed (or `N/A` for headless phases).

### Known Issues & Deferred Items
- Any non-blocking limitation explicitly deferred to a later phase (or `None`).
```

### 10.3 Changelog rules

- **Never delete or falsify history.** Do not edit past completed release sections except to fix a typo. Do not record placeholder numbers—run the commands and record actual test counts and timings.
- **Reference requirement IDs (`R1`–`R11`)** so anyone reviewing the project can trace every requirement from `agents.md` to code, tests and changelog entries.
- **Always read and write with `encoding="utf-8"`.**

---

## 11. Common pitfalls and defensive checklist

Keep these project-specific traps in mind during implementation:

1. **`previous_chapters.py` imports `matplotlib.pyplot` at module top level.**
   - Never import `previous_chapters.py` inside `core`, `narration` or `render`.
   - Inside `probe/loading/code_import.py`, set `os.environ.setdefault("MPLBACKEND", "Agg")` before executing user model code so Tk/Qt GUI backends are never initialized during CLI recording or headless tests.
2. **`pygame` vs `pygame-ce` collision.**
   - Both packages provide `import pygame`. Installing both corrupts the namespace. Always specify `pygame-ce` in `pyproject.toml`, and have `tilevis check` inspect `importlib.metadata` to warn if legacy `pygame` is also present.
3. **Windows lecture-hall environment traps.**
   - **Encoding:** Korean Windows defaults `open()` and `subprocess` text streams to `cp949`. Always pass `encoding="utf-8"` explicitly.
   - **DPI scaling:** Windows laptops/lecterns commonly run at 125% or 150% scale. Call DPI awareness in `render/platform.py` *before* `pygame.init()` / `pygame.display.init()`.
   - **Screenshare black screen:** Hardware exclusive `FULLSCREEN` often captures as a black rectangle or flickers on `Alt+Tab` in Zoom/Teams. Default to `borderless` (`NOFRAME` at desktop size).
   - **Stuck modifier keys on `Alt+Tab`:** Clear internal modifier tracking or read `pygame.key.get_mods()` fresh on `WINDOWFOCUSGAINED`.
4. **Weight matrix orientation (`nn.Linear`).**
   - PyTorch stores `nn.Linear.weight` as `(out_features, in_features)`. Always transpose to `(in_features, out_features)` in `probe/weights.py` so `input (T, in) @ weight (in, out) = output (T, out)` matches the visual left-to-right matmul scene.
5. **Modules called multiple times per forward pass.**
   - In `previous_chapters.TransformerBlock`, `self.drop_resid` is a single `nn.Dropout` instance called **twice** in `forward()`. Always key captures by `(module_path, call_index, io)` rather than `module_path` alone.
6. **GPT-2 activation outliers.**
   - Trained GPT-2 models develop a handful of dimensions with magnitudes 10×–50× larger than the rest. Scaling colors by `abs_max` washes 99% of tiles to pale grey/white. Use `abs_p99` (with clamping to `[-1, 1]`) in `per_tensor` and `shared_step` normalizers so structure remains vivid, while tooltips still show the exact unclipped number.
7. **Single-token prompts and `n_heads` inference.**
   - When `seq_len == 1`, causal self-attention softmax over a single key is `[1.0]` regardless of `n_heads`, so `context_vec` equals `values` for *every* divisor of `emb_dim`. Detect `T == 1`, mark fidelity `unverified`, and warn the user to supply at least 2 tokens when `n_heads` is not specified in the profile.

---

## 12. Out of scope and future extension paths

### 12.1 Explicitly out of scope for `1.0.0`

- **Direct TensorFlow checkpoint loading (`gpt_download.py`).** `gpt_download.py` requires `tensorflow`, which is heavy and fragile across Python versions. Teammates already save finetuned models as PyTorch `.pth` files (`torch.save(model.state_dict(), ...)`). If someone wants to visualize raw pretrained GPT-2 weights before finetuning, convert them once in the `ch07` training environment with `torch.save(model.state_dict(), "gpt2_pretrained.pth")` and point a `tilevis` profile at that `.pth`.
- **Threaded mid-forward-pass live stepping.** Pausing PyTorch inside `forward()` adds thread synchronization hazards without visual benefit for inference demos. Recording a `Trace` takes seconds and guarantees deterministic backward/forward stepping and torch-free lecture-hall playback.
- **Training / backpropagation visualization.** `1.0.0` visualizes inference forward passes only.

### 12.2 Designed-in extension seams (post-`1.0.0`)

Because every seam is backed by a `Registry` and the versioned `Trace` contract, future contributors can add capabilities without refactoring existing modules:
- **New model architectures (Llama-style RoPE, RMSNorm, SwiGLU, GQA):** register a new `ArchitectureAdapter` in `probe/adapters/`, add `RMSNorm` to `WEIGHT_EXTRACTORS`, and optionally register a `StepBuilder` or reuse existing `LAYERNORM` / `LINEAR` / `ATTENTION_PATTERN` kinds.
- **Korean or multilingual captions:** drop `ko.toml` into `src/tilevis/narration/locales/`, bundle Noto Sans KR in `assets/fonts/`, and set `locale = "ko"` in `presenter/default.toml`.
- **Side-by-side checkpoint comparison ("before vs after finetuning"):** load two `Trace` files recorded on the same prompt and add a diff mode to `App` comparing corresponding `Step.id`s.
- **Web or notebook player:** read `*.trace.npz` directly in Python/JS (it is standard ZIP + UTF-8 JSON + raw `float32` `.npy` arrays) without touching `probe`.
