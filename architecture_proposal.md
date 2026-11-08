# LLM Tile Visualizer: Architecture Proposal

## 1. Goals

| Goal | Design consequence |
|---|---|
| Show weights and values of each node as colored tiles | A tensor-to-tile-grid mapping layer and a color mapper |
| Step manually through the LLM's calculation | Pre-recorded "timeline" of steps with a player (next/back/jump) |
| Show how input vectors are changed by each layer's weights | Every step records input, weights and output together |
| Audience with little LLM knowledge, intuitive and fun | A "friendly names" layer, few details, animation, optional "zoom" |
| Sits on top of `previous_chapters.py` | Non-invasive hooks, with **no edits to the original file** |
| Two main modules: hook/watch and render | Strict separation, connected only by a plain-data interface |
| Modular and expandable | Plugin-style registries for layer handlers, color maps and scenes |

## 2. Findings about the existing model

From `ch07/01_main-chapter-code/previous_chapters.py`:

```
GPTModel
 ├─ tok_emb, pos_emb (nn.Embedding) → add → drop_emb
 ├─ trf_blocks (nn.Sequential of N × TransformerBlock)
 │    ├─ norm1 (LayerNorm) → att (MultiHeadAttention) → dropout → + shortcut
 │    └─ norm2 (LayerNorm) → ff (Linear → GELU → Linear) → dropout → + shortcut
 ├─ final_norm (LayerNorm)
 └─ out_head (Linear → vocab logits)
```

Two things shape the design:

1. **Module forward hooks see only module boundaries.** `nn.Linear`, `LayerNorm`, `Embedding` and `GELU` are modules, so a hook can capture them.
   - The residual additions (`x + shortcut`) happen inside `TransformerBlock.forward`.
   - The attention internals (scores, mask, softmax, weighted sum) are plain tensor ops inside `MultiHeadAttention.forward`.
   - Hooks cannot see either of these. The attention internals are the most "fun" and explainable part, so they need a dedicated approach (section 4.3).
2. **GPT-2 is huge.** GPT-2 medium has 1024-dim embeddings, a 50k vocab and 24 layers. Rendering every value as a tile is impossible, so there must be a downsampling or summarizing stage. For the demo, the recommended plan is to use a **small config** (for example `emb_dim=48`, 2 to 4 layers, 4 heads) and a short prompt of 4 to 6 tokens.

## 3. High-level architecture

```
┌────────────────────┐    ┌───────────────────────┐    ┌─────────────────────┐
│  Original program  │    │  Module A: PROBE      │    │  Module B: RENDERER │
│ previous_chapters  │───►│  hooks + recorder     │───►│  tiles + UI         │
│  (unmodified)      │    │  (torch-aware)        │    │  (torch-free)       │
└────────────────────┘    └──────────┬────────────┘    └──────────▲──────────┘
                                     │  Trace (plain numpy data)  │
                                     └───────►  core/  ◄──────────┘
                                         (shared data contract)
                                                 │
                                       ┌─────────┴─────────┐
                                       │ Player/Controller │  step forward / back
                                       └───────────────────┘
```

**Key idea:** the probe runs one forward pass and records everything into a **`Trace`**, an ordered list of `Step` objects, each holding plain numpy arrays. The renderer only ever reads a `Trace`.

Benefits:
- **Manual stepping is trivial.** Step back and forward is an index into a list, with no need to pause PyTorch mid-computation or use threads.
- **The renderer never imports torch.** It can be tested, replaced or developed with fake data.
- **Traces can be saved to disk** (`.npz`/pickle). The presentation then runs instantly, with no model loading on stage. This is the main safety net for a live presentation.
- **Alternative frontends** (pygame, matplotlib, web) can reuse the same trace.

> [!NOTE]
> Live mode, where the model streams steps into the renderer as they happen, is a later extension. The same `Step` objects would be pushed through a queue.

## 4. Proposed directory layout

```
llm_tile_visualizer/
├── ch07/                         # existing, untouched
├── architecture_proposal.md
├── tilevis/
│   ├── core/                     # shared contract, no torch or pygame imports
│   │   ├── trace.py              # Trace, Step, TensorSnapshot dataclasses
│   │   ├── events.py             # optional event bus (for live mode later)
│   │   └── config.py             # settings (tile size, theme, downsample limits)
│   │
│   ├── probe/                    # MODULE A: hooks into the LLM (torch lives here)
│   │   ├── recorder.py           # TraceRecorder: attach, run, detach, build Trace
│   │   ├── hooks.py              # generic forward-hook helpers for nn.Module
│   │   ├── handlers/             # one handler per layer type (registry)
│   │   │   ├── base.py           # LayerHandler interface
│   │   │   ├── embedding.py
│   │   │   ├── linear.py
│   │   │   ├── layernorm.py
│   │   │   ├── activation.py     # GELU
│   │   │   ├── attention.py      # re-computes q, k, v, scores, softmax
│   │   │   └── residual.py       # shortcut additions
│   │   ├── model_loader.py       # imports previous_chapters, builds a small demo GPT
│   │   └── summarize.py          # downsample, pool, select top-k values
│   │
│   ├── render/                   # MODULE B: draws the Trace (no torch)
│   │   ├── app.py                # window, main loop, event dispatch
│   │   ├── player.py             # StepPlayer: cursor, next/prev/jump, autoplay
│   │   ├── colormap.py           # value → color (diverging, themes, registry)
│   │   ├── tiles.py              # TileGrid: array → grid of tiles (+ layout)
│   │   ├── scenes/               # one scene per Step kind (registry)
│   │   │   ├── base.py           # Scene interface
│   │   │   ├── tokens.py         # text → token IDs → embedding vectors
│   │   │   ├── matmul.py         # input tiles × weight tiles → output tiles
│   │   │   ├── attention.py      # who-looks-at-whom heatmap
│   │   │   ├── residual.py       # add-back animation
│   │   │   └── output.py         # next-word probabilities (bar chart)
│   │   ├── widgets/              # HUD: step counter, caption box, legend, buttons
│   │   ├── animation.py          # tweening / easing for transitions
│   │   └── themes/               # color palettes and fonts
│   │
│   ├── narration/
│   │   └── captions.py           # kind → plain-English caption, with a "nerd mode" toggle
│   │
│   └── cli.py                    # entry points: record / play / live
│
├── traces/                       # saved .npz traces (e.g. demo.npz)
├── tests/                        # unit tests (fake traces, hook tests)
├── requirements.txt
└── run_demo.py                   # `python run_demo.py --prompt "Every effort moves you"`
```

### 4.1 Shared data contract (`core/trace.py`)

```python
@dataclass
class TensorSnapshot:
    name: str                 # "block0.att.W_query.weight"
    role: str                 # "input" | "weight" | "output" | "attn_weights" ...
    data: np.ndarray          # already reduced to 1D/2D for display
    full_shape: tuple         # original shape, for labels
    stats: dict               # min, max, mean, std (for color scaling)

@dataclass
class Step:
    id: int
    kind: str                 # "embed" | "linear" | "layernorm" | "attention" | "residual" | ...
    layer_path: str           # "trf_blocks.0.att"
    block_index: int | None
    title: str                # friendly name, "Block 1 · Paying attention"
    inputs: list[TensorSnapshot]
    weights: list[TensorSnapshot]
    outputs: list[TensorSnapshot]
    meta: dict                # tokens, head index, etc.

@dataclass
class Trace:
    tokens: list[str]
    steps: list[Step]
    model_info: dict          # n_layers, emb_dim, ...
```

This is the **only** interface between the two modules. If it stays stable, each module can change freely.

### 4.2 Module A: Probe (hook and watch)

- **`TraceRecorder`**: a context manager. It registers forward hooks on the selected modules, runs `model(input_ids)` under `torch.no_grad()`, collects `Step`s in execution order, and removes the hooks on exit. `previous_chapters.py` is never modified.
- **Layer handlers** are registered by layer type (`@register(nn.Linear)`). A handler turns `(module, inputs, output)` into a `Step`: which weights to show, and which tensors count as input and output. Supporting a new layer type means adding one file.
- **Naming:** hooks use `named_modules()` paths, so steps are tagged with their block index and role (`att.W_query`, `ff.layers.0`, etc.).
- **Summarizer (`summarize.py`):** reduces big tensors to something displayable by taking the first N tokens, a sampled or pooled slice of the embedding dimension, and recording stats for stable color scaling. The reduction is configurable and deterministic, so a tile always means the same thing across steps.

### 4.3 Capturing what hooks cannot see

| Hidden detail | Strategy |
|---|---|
| Attention scores, mask and softmax | The `attention` handler **re-computes** them from the hooked q/k/v projections (the same few lines as the original `forward`). It runs on the captured input, so the result is identical and the source is untouched. A check compares the recomputed output to the real one. |
| Residual shortcut (`x + shortcut`) | Hook `norm1`/`norm2` inputs (the shortcut value) and the `drop_resid` output to derive the residual step: `out = shortcut + sublayer_out`. |
| Dropout | Inactive in `eval()` mode, so it is skipped. |

> [!IMPORTANT]
> Fallback if recomputing proves fragile: ship a thin **subclass** of `MultiHeadAttention` in `probe/` that stores its intermediates (`self.last_scores`, `self.last_weights`). It would be swapped in at load time, still without editing `previous_chapters.py`. This option is also viable as the primary approach. It is decided during implementation.

### 4.4 Module B: Renderer

- **Library: `pygame`** (recommended). It gives smooth animation, easy keyboard and clicker input, fullscreen output for a projector, and simple drawing of thousands of colored rectangles. Alternatives are `matplotlib` (simple, but animation and stepping UI are clunky) and a web frontend (flexible, but adds a JS stack).
- **`StepPlayer`**: holds the `Trace` and a cursor.
  - Controls: `→` / Space next, `←` back, `Home`/`End`, number-key jump to a block, `N` toggle nerd mode, `F` fullscreen, `Esc` quit.
  - Supports presentation clickers, which usually send PageUp/PageDown.
- **`ColorMap`**: maps a value to a color. Defaults to a **diverging palette** (blue = negative, white = zero, orange/red = positive) so weights and activations read at a glance. Colors are normalized with a per-step or global scale (configurable). Themes are swappable.
- **`TileGrid`**: turns a 1D or 2D array into a grid of tiles with a computed layout. It handles hover tooltips (the real number) and highlight states.
- **Scenes**: one class per `Step.kind`, chosen via a registry. A scene decides *how* a step is drawn. The **matmul scene** is the core of "how input vectors are manipulated by weights":
  1. The input vector appears as a row of tiles.
  2. The weight matrix appears as a tile grid.
  3. On each press, one output tile is "computed" (its row and column highlight, then the output tile fades in), or all of them animate together.
  4. The result is shown next to the input, so the change is visible.
- **Animation** uses a simple tween engine (ease-in/out, pulses, "flying" tiles), kept in one module so the fun factor can be tuned independently.
- **Layout idea:** the main stage in the center, a small "pipeline minimap" at the top showing progress through the whole model (embedding → blocks → output), and a caption box at the bottom.

### 4.5 Narration (audience-friendly layer)

`captions.py` maps each step kind to a short, friendly caption, for example:

- embed: "Each word becomes a list of numbers: its *fingerprint*."
- attention: "Words look at each other to figure out what matters."
- linear: "The model *mixes* the numbers using its learned weights."
- layernorm: "Squash the numbers so none get too loud."
- residual: "Keep a copy of what we had, then add the new insight."
- output: "Pick the most likely next word."

A **nerd mode** toggle shows the real layer names and tensor shapes. The default stays minimal.

## 5. Data flow

```
prompt ─► tokenizer ─► input_ids
                          │
 (probe) TraceRecorder: attach hooks ─► model.eval(); model(input_ids)
                          │               hooks fire in execution order
                          ▼
              handlers build Step objects ─► summarize/downsample
                          ▼
                  Trace  ──(optional)──► traces/demo.npz
                          ▼
 (render) StepPlayer(trace) ─► Scene[kind].draw(step) ─► viewport
              ▲ keyboard / clicker input
```

## 6. Extensibility points

| To add... | Do this |
|---|---|
| A new layer type | Add a `probe/handlers/*.py` file with `@register(...)` |
| A new visual for a step | Add a `render/scenes/*.py` file registered for a `Step.kind` |
| A new color theme | Add a palette to `render/themes/` |
| A different model | Provide a new `model_loader`. The recorder is generic. |
| A different frontend | Consume `Trace` (for example a Jupyter or web frontend) |
| Live streaming | Push `Step`s through `core/events.py` instead of building the full `Trace` first |
| Training visualization (weights changing over epochs) | Record multiple traces and add a "diff" scene |

## 7. Technical decisions and risks

| Item | Decision / mitigation |
|---|---|
| Model size | Use a small GPT config for the visuals, trained or loaded as a mini model. Optionally show the real GPT-2 using pooled summaries. |
| Meaningful output | A randomly initialized model produces gibberish. For the presentation, use the project's trained or finetuned weights (`load_weights_into_gpt`, or a `.pth` checkpoint). |
| Color scaling | Per-step normalization makes weights and activations comparable inside a step. A global scale option avoids misleading jumps between steps. |
| Performance | The recorded trace is plain arrays, and only the visible step is drawn. Tile surfaces are cached. |
| Presentation safety | Pre-recorded traces mean no heavy model load or PyTorch dependency on the presentation machine. |
| Testing | Unit tests with a tiny model (recomputed attention matches the real output) and with fake traces for the renderer. |
| Dependencies | `torch`, `numpy`, `tiktoken` (already used), plus `pygame`. |

## 8. Suggested implementation phases

1. **Core and probe:** `Trace`/`Step`, `TraceRecorder`, the linear/embedding/layernorm handlers, and a console dump to verify the trace.
2. **Minimal renderer:** window, `StepPlayer`, `ColorMap`, a static `TileGrid` scene, and next/back keys.
3. **The key scene:** the matmul animation (input × weights → output).
4. **Attention and residual:** the attention handler and scene, then the residual scene.
5. **Fun and polish:** captions, themes, the minimap, transitions, tooltips, nerd mode, trace save/load, and fullscreen.
6. **Stretch goals:** live mode, next-word probability chart, and training-progress playback.

## 9. Open questions

1. **Frontend library:** is `pygame` fine, or do you prefer matplotlib or a browser-based UI?
2. **Model for the demo:** a small custom-sized GPT, the pretrained GPT-2 (pooled), or your finetuned checkpoint?
3. **Step granularity:** should one press compute the *whole layer* at once, or reveal it *one output node at a time*?
4. **Presentation setup:** do you have a projector and resolution requirement, and will you use a clicker or just the keyboard?
5. **Scope:** do you want the final next-word prediction shown, or only the internal layers?
