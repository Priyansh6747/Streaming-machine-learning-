# PDF Generation Skill — Master Plan (v2 — Graphics Expansion)

> **v2 goal:** a skill that lets **any** agent (even a weak/small model) produce
> visually rich, study-ready PDFs — text hierarchy *plus* charts, diagrams, math
> formulas, syntax-highlighted code — by **predefining all rendering code** up front.
> The agent passes data; the skill renders. Like `dsa-judge`, the skill owns
> deterministic tooling; the agent owns reasoning & content.
>
> v1 was text-only (paragraphs, bullets, tables, code blocks, callouts, page numbers).
> v2 adds: **matplotlib charts**, **graphviz diagrams** (flowcharts / trees / state
> machines / block diagrams), **matplotlib-mathtext formulas**, and **pygments code
> highlighting** — all behind the same `kit.add_*` notes-as-code API so a weak model
> just calls one function and gets back a polished graphic.

---

## 0. Glossary & Conventions

- **Agent** = the LLM driver using the skill.
- **Skill** = `notes_kit.py` + `graphics_kit.py` + `highlighting.py` + `runner.py`.
- **Kit** = the public helper library (`add_heading`, `add_bar_chart`, `add_flowchart`,
  `add_code`, ...). Accessed by the agent as `kit.*` (the runner passes the importable
  `notes_kit` module as `kit`).
- **Story** = the ordered list of flowables ReportLab renders top-to-bottom.
- **Flowable** = any renderable unit (paragraph, table, image, spacer, divider, page break).
- **Asset** = a PNG/SVG rendered by a graphics helper, stored in the run's `assets/` dir,
  then embedded into the story via ReportLab's `Image` flowable. The agent never sees the
  asset file — `add_bar_chart` returns nothing; the chart appears in the PDF as if by magic.
- **Run folder** = `workspace/run<id>/` per generation, holding:
  - `content.py`     — agent-authored notes-as-code
  - `notes.pdf`      — produced artifact
  - `preview_p1.png` — validation screenshots (first 2 pages)
  - `assets/`        — chart/diagram/math PNGs rendered for this run
- **Notes-as-code paradigm:** notes are a *sequence of function calls*. The agent never
  freehands layout, never imports matplotlib/graphviz/reportlab, never computes coordinates.

### Hard rules (v1, retained)

1. **Base14 fonts by default** (Helvetica/Helvetica-Bold/Courier) so it works offline.
   Custom `.ttf` registration is opt-in via `PDF_SKILL_FONTS_DIR`.
2. **Never use Unicode sub/super glyphs** (₀ ²). Use `<sub>`/`<super>` inline tags, or
   `add_math()` for display-size formulas.
3. **Escape `&`, `<`, `>` before inserting raw code into a `Paragraph`** (handled by kit).
4. **Always visually validate** first 1–2 pages before presenting the PDF.
5. Tables don't accept raw strings for rich content; pass `Paragraph` flowables per cell.

### Hard rules (v2, new — graphics)

6. **Graphics helpers never raise** on bad/missing input — they either degrade gracefully
   (e.g., a "missing data" placeholder box) or surface the issue via the build JSON's
   `warnings[]`. The PDF always renders.
7. **The agent passes only data** — values, edges, labels. The kit owns all styling
   (palette, fonts, sizes, line weights, layout algorithm). No `color=`, `linewidth=`,
   `figsize=` arguments are exposed at the helper API; agent-facing kwargs are limited to
   `title`, `caption`, `xlabel`, `ylabel`, `width`, `kind`, `highlight`.
8. **Charts render to transparent PNG** so they look correct on either light or dark
   theme. DPI = 200 for charts (crisp at print, reasonable file size); 300 for math.
9. **Code highlighting is opt-in** via `highlight=True` on `add_code`. Default stays the
   existing dark-monospace block (no surprises, keeps zero-dep base path working).
10. **Diagrams require the system `dot` binary** (`apt install graphviz`). If absent,
    `add_flowchart` / `add_tree` / `add_state_diagram` / `add_block_diagram` / `add_graph`
    emit a warning in build JSON and insert a styled placeholder telling the agent to fall
    back to a numbered list or table. Charts (matplotlib) never need a system dep.
11. **Asset files live in `workspace/run<N>/assets/`** and are *reused across rebuilds* of
    the same run (keyed by a stable hash of helper name + args). This avoids re-rendering
    identical charts on every build cycle — critical for agent iteration speed.
12. **All graphics helpers accept an optional `caption`**; the kit renders it in the
    `Caption` style (muted, centered) directly under the image.

---

## 1. High-Level Architecture (v2)

```
PDF_Skill/
├── SKILL.md               # Standard skill metadata (name, description, license, triggers)
├── plan.md                # This file (architecture + step-by-step recipe with graphics).
├── func.md                # Snippet library the agent copies from verbatim.
├── pyproject.toml         # Dependencies pinned (charts, diagrams, highlighting added in v2).
├── runner.py              # CLI: init, build, validate (unchanged surface, expanded validators).
├── notes_kit.py           # Core: design tokens, styles, text add_* helpers, NotesCanvas,
│                          #       generate_notes_pdf, AND re-exports everything from
│                          #       graphics_kit.py + highlighting.py so `from notes_kit import *`
│                          #       stays the single import surface for the agent.
├── graphics_kit.py        # NEW: charts (matplotlib), diagrams (graphviz), math (matplotlib).
├── highlighting.py        # NEW: pygments-based code formatter → ReportLab Paragraph XML.
├── schemas/
│   ├── __init__.py
│   └── generate_spec.py    # Pydantic request/response models.
├── workspace/             # sandbox; one subdir per run. Created by runner.
│   └── run<id>/
│       ├── content.py     # agent-authored notes-as-code
│       ├── notes.pdf      # produced artifact
│       ├── preview_p1.png # validation screenshots
│       └── assets/        # rendered chart/diagram/math PNGs (hashed cache)
└── main.py                # legacy; unused by skill.
```

### Boundary (unchanged from v1, restated for graphics)

- **Agent owns:** choosing which helper to call, supplying the *data* (numbers, edges,
  labels), textual content, running `runner.py`, parsing JSON, iterating on fixes from
  validation feedback.
- **Skill owns:** all rendering — design tokens, font registration, paragraph styles,
  page footer, chart themes, diagram layout, code highlighting, asset caching, layout
  validation.

The agent **never** imports `matplotlib`, `graphviz`, `pygments`, or `reportlab` directly
in `content.py`. It only calls `kit.add_*` helpers.

---

## 2. Skill Contract (Agent ⇄ Skill)

Surface unchanged from v1. Same `init` / `build` / `validate` subcommands; same JSON
shapes. v2 only adds extra optional kwargs to `build` and new fields inside `BuildResponse`.

### 2.1 Build command

```
uv run python runner.py build \
    --content workspace/run1/content.py \
    --out    workspace/run1/notes.pdf \
    --title  "Dynamic Programming" \
    --subtitle "Core patterns" \
    --preview
```

### 2.2 BuildResponse JSON (additions marked)

```json
{
  "success": true,
  "pdf_path": "workspace/run1/notes.pdf",
  "pages": 4,
  "previews": ["workspace/run1/preview_p1.png", "workspace/run1/preview_p2.png"],
  "assets":   ["workspace/run1/assets/bar_chart_a1b2.png", "workspace/run1/assets/tree_c3d4.png"],
  "warnings": ["graphviz 'dot' binary not found; add_flowchart on page 2 used a placeholder."],
  "errors":   []
}
```

- `assets[]` — NEW. Lists every chart/diagram/math image embedded in this build. Useful
  for debugging ("the agent can `view` these directly if needed").
- `warnings[]` — NEW use: graphics-related degradations (missing graphviz binary, empty
  data, single-value pie chart, etc.). Do NOT block the build; agent MAY act on them.

### 2.3 ValidateResponse JSON (additions marked)

```json
{
  "success": true,
  "issues": [
    {"kind": "image_oversized", "page": 2, "severity": "error",
     "detail": "Chart on page 2 (a1b2.png) is wider than content frame by 12pt."},
    {"kind": "low_contrast_image", "page": 3, "severity": "warn",
     "detail": "Image on page 3 has <5% dark pixels; likely blank/failed render."},
    {"kind": "orphan_caption", "page": 4, "severity": "warn",
     "detail": "Caption 'Figure 3' is more than 4pt from preceding image — may be detached."}
  ]
}
```

Three new issue kinds for graphics — see §5.

---

## 3. Modules — Detailed Responsibilities

### 3.1 `notes_kit.py` (core — slight v2 changes)

Same as v1, plus:
- At end of file: `from graphics_kit import *  # noqa` and `from highlighting import *  # noqa`
  so every graphics helper is reachable as `notes_kit.add_bar_chart` (matches how the
  runner passes `kit = notes_kit` to agent code).
- Extend `__all__` to include graphics helper names.
- Use sentence-case section banner comments so the file is navigable:
  `# === 5. Text helpers ===`, `# === 6. Graphics re-exports ===`, etc.

### 3.2 `graphics_kit.py` (NEW — the v2 centerpiece)

Single self-contained module holding every chart/diagram/math/code-highlight helper.
Order of sections:
1. **Imports** — lazy: matplotlib, graphviz (python wrapper), numpy, math, hashlib, os.
2. **Module-level theme injection** — read palette + fonts from `notes_kit` so charts use
   the same brand colors. NO local palette redefinition; import `INK`, `ACCENT`, etc.
3. **Asset cache** (`_asset(path, hash_fn, render_fn)`) — see §11.2.
4. **Generic chart wrapper** `_render_chart(story, render_fn, caption, width)` — does
   dpi/size/transparent-bg/saving/embedding/caption.
5. **Chart helpers** — `add_bar_chart`, `add_line_chart`, `add_pie_chart`, `add_donut_chart`,
   `add_scatter`, `add_histogram`, `add_box_plot`, `add_area_chart`, `add_heatmap`.
6. **Diagram layer** `_render_dot(story, dot_src, caption, width)` — calls `dot -Tpng`,
   embeds, gracefully degrades if binary missing.
7. **Diagram helpers** — `add_flowchart`, `add_tree`, `add_state_diagram`,
   `add_block_diagram`, `add_graph`, `add_call_graph`, `add_venn`, `add_timeline`.
8. **Math helper** `add_math(story, expr, display=True, width=None, caption=None)`.
9. **Highlighting hook** — none here; `add_code` highlighting lives in `highlighting.py`.

Hard requirements:
- Zero side effects on import (only definitions + closing the matplotlib `Agg` backend).
- All public helpers take `story` first, mutate it in place, return `None`.
- All public helpers accept an optional trailing `caption=` and `width=`.
- All public helpers degrade gracefully per §9 (never raise).

### 3.3 `highlighting.py` (NEW)

Bundle that upgrades `add_code` from plain monospace to syntax-colored highlighted code.

- Public function: `highlight_to_reportlab(code, language, style_name="monokai")`
  → returns a `Paragraph` instance whose text is ReportLab-XML with `<font color="...">`
  spans per token (or a `Table` of per-line `Paragraph`s for line numbers + alignment).
- Uses `pygments` with a tiny custom ReportLab formatter (`PygmentsToReportlabFormatter`).
  Built-in formatters output HTML/Latex/RTF — none native to reportlab, so we write one
  (~50 LOC) that walks tokens and emits `<font color="#hex">tok</font>`.
- Style: locked to **monokai** by default (looks great on `CODE_BG` #0F172A). Exposed as
  `kit.add_code(story, code, language="python", highlight=True)` — agent toggles only.
- Falls back to plain escaped code (v1 behavior) if pygments import fails or `language`
  is unknown to pygments.
- Optional `line_numbers=True` on `add_code` (defaults off — keeps code compact).

### 3.4 `runner.py` (mostly unchanged; build pipeline tweaks)

- Build pipeline calls `notes_kit.generate_notes_pdf(...)`.
- Page-count via `pypdf`. Previews via `pdf2image`. As before.
- AFTER build: scan the run's `assets/` dir, list every file, expose in `assets[]` in JSON.
- Validate: 3 new heuristics for graphics (see §5).
- New optional CLI flag: `--no-cache` to wipe `assets/` before building (for safety when
  the agent suspects stale assets). Default keeps cache.

### 3.5 `schemas/generate_spec.py`

Add to `BuildResponse` a field `assets: list[str] = []`. No other model changes.

### 3.6 `SKILL.md`

Update the description to mention charts/diagrams/math/highlighting-capability.
Add a "Graphics helpers" table pointing at `func.md` §21–§30 (the new snippet sections).

---

## 4. Agent Workflow (End-to-End Recipe, v2)

Same 9 steps as v1; graphics are integrated as just another set of helpers in step 3:

1. **Plan the outline**. When a section benefits from a chart or diagram, mark it inline
   in the outline as `[chart: bar, complexity vs n]` or `[diagram: tree, recursion]`.
2. `uv run python runner.py init --name dp_notes` → scaffold.
3. Open `func.md` and copy the snippet for each outline block. For a chart/diagram, copy
   the matching snippet from §21 (charts) or §22 (diagrams). **Edit only the data**.
4. Write `content.py` as a sequence of `kit.add_*` calls. Mix text and graphics freely;
   `add_keep_together` works across text and graphics alike.
5. Build: `uv run python runner.py build --content ... --out ... --title ... --preview`.
6. Parse JSON. On failure, fix errors. On success, view the previews.
7. Validate: `uv run python runner.py validate --pdf ...`.
8. Fix any `severity: error` issue (new types in v2: `image_oversized`, `low_contrast_image`).
9. Iterate until clean. Present final path to user.

### 4.1 Example content.py with graphics (the agent's whole job)

```python
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_title_block(story, "Dynamic Programming",
                 "Patterns, complexity, and recursion trees")

    kit.add_heading(story, "Recognizing DP", level=1)
    kit.add_paragraph(story,
        "DP overlaps subproblems; memoizing collapses exponential trees into linear work.")

    kit.add_heading(story, "Recursion tree for fib(5)", level=2)
    kit.add_tree(story,
        edges=[
            ("fib(5)", "fib(4)"),
            ("fib(5)", "fib(3)"),
            ("fib(4)", "fib(3)"),
            ("fib(4)", "fib(2)"),
            ("fib(3)", "fib(2)"),
            ("fib(3)", "fib(1)"),
        ],
        caption="Figure 1 — overlapping subproblems in naive recursion")

    kit.add_heading(story, "Time complexity", level=2)
    kit.add_bar_chart(story,
        title="Calls vs n (naive vs memoized)",
        categories=["n=1", "n=2", "n=3", "n=4", "n=5", "n=6"],
        series=[
            {"label": "Naive", "values": [1, 2, 4, 8, 15, 25]},
            {"label": "Memo",  "values": [1, 2, 3, 4, 5, 6]},
        ],
        ylabel="call count",
        caption="Figure 2 — exponential vs linear growth")

    kit.add_callout(story,
        "Memoization makes the tree a DAG — every subproblem solved once.",
        kind="tip")

    kit.add_heading(story, "Closed form", level=2)
    kit.add_math(story,
        r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}",
        caption="Figure 3 — arithmetic series")

    kit.add_code(story,
        "def fib(n, memo={}):\n    if n <= 1: return n\n    "
        "if n not in memo:\n        memo[n] = fib(n-1)+fib(n-2)\n    return memo[n]",
        language="python", highlight=True, line_numbers=True)
```

The agent only fills data strings (and `highlight=True`/`line_numbers=True` flags). Every
other styling decision is the kit's responsibility.

### 4.2 Repair loop heuristics — graphics additions

| Issue kind              | How the agent fixes it in content.py                         |
|-------------------------|--------------------------------------------------------------|
| `image_oversized`       | Pass a smaller `width=` on the offending helper (e.g., `150*mm`). |
| `low_contrast_image`    | Likely the renderer returned an empty/blank PNG. Rerun build (`--no-cache` if stale), or replace the helper call with an equivalent `add_table` / `add_bullets` block. |
| `orphan_caption`        | Wrap the graphic + its caption in `kit.add_keep_together(..., lambda s: kit.add_bar_chart(s, ...), lambda s: ...)` — already default behavior, so usually means the caption is detached from a preceding image by manual code; consolidate the calls. |
| Warning: `dot missing`  | Replace the `add_flowchart`/`add_tree`/etc. with stepwise `add_numbered_list` of the same edges, or document the system dep gap to the user (do not try to install from content.py). |
| Warning: `pygments unknown language` | Drop the `highlight=True` flag from `add_code` (keep `language="..."` if you want the small uppercase label). |

---

## 5. Validation Heuristics (skill side)

v1 rules kept (orphan_heading, code_overflow via reportlab-frame check, text_touches_margin,
overflow, empty_page). v2 adds three graphics-specific heuristics, all implemented in
`runner.py validate` using `pdfplumber` for word/object positions on each rendered page:

### 5.1 image_oversized (error)

Any `Image` flowable on the page whose embedded image's drawn width exceeds
`PAGE_SIZE[0] - 2*MARGIN + 1pt`. We detect by inspecting `page.images` in pdfplumber and
comparing each image's bbox right edge to the content-frame right edge.

### 5.2 low_contrast_image (warn)

For each image on the page, count the fraction of "dark" pixels (luminance < 0.5 on a 0..1
scale). Use `pdf2image` to already-rendered previews OR call `PIL.Image.open(asset_path)`
directly (we know the asset paths from the build JSON or from `assets/`). If <5% of pixels
are non-white AND the image occupies >30% of the page area, flag as likely-blank render.
Cheap heuristic (no perfect detection needed).

### 5.3 orphan_caption (warn)

A `Caption`-styled paragraph whose top Y is more than 4pt below the bottom of any preceding
flowable on the page (so the caption is detached from its image). Cheap: walk
`page.extract_words()` and find caption-like short centered strings, measure gap.

### 5.4 Existing rules, kept

- `orphan_heading`, `code_overflow`, `text_touches_margin`, `overflow`, `empty_page` — see
  v1 plan §5. Unchanged in implementation.

---

## 6. Design Tokens (locked palette — extended for charts)

All v1 colors retained. v2 adds a **chart palette** — a rotation of distinct hues used
when a chart has multiple series and no agent-supplied colors. Charts by default pick from
this rotation for series 1..N.

```python
CHART_PALETTE = [
    "#2563EB",   # ACCENT  — series 1
    "#FF8A3D",   # ACCENT_2 — series 2
    "#059669",   # emerald — series 3
    "#D97706",   # amber  — series 4
    "#7C3AED",   # violet — series 5
    "#DC2626",   # red    — series 6 (used sparingly)
    "#0891B2",   # cyan   — series 7
    "#475569",   # slate  — series 8
]
```

Chart styling tokens:

| Token              | Value    | Used for                              |
|--------------------|----------|---------------------------------------|
| CHART_BG           | "none" (transparent) | chart figure background     |
| CHART_TEXT_COLOR   | INK `#1A1A1A` | axis labels, tick labels         |
| CHART_TITLE_COLOR  | HEADING `#0F172A` | chart titles                  |
| CHART_GRID_COLOR   | `#E2E8F0` | axis gridlines (very light)        |
| CHART_TICK_COLOR   | MUTED `#64748B` | axis tick marks                |
| CHART_EDGE_COLOR   | BORDER `#CBD5E1` | bar/line outlines             |
| CHART_DPI          | 200      | raster resolution for embedded PNGs  |
| MATH_DPI           | 300      | math formula PNG resolution          |
| CHART_DEFAULT_W    | 170*mm   | default chart width (full frame on A4) |
| DIAGRAM_DEFAULT_W  | 170*mm   | default diagram width                |
| MATH_DEFAULT_W     | None     | auto-fit to expression               |

The kit MUST use these tokens so charts remain on-brand even if the agent passes messy
data. The agent has NO access to color arguments at the helper API.

---

## 7. Type Hierarchy (unchanged) — see v1 plan §7

---

## 8. Build Pipeline Internals (v2 — updated steps)

Sequential inside `runner.py build`:

1. Sanity checks on `--content` and `--out` paths (as v1).
2. Inject repo root into `sys.path` so `from notes_kit import *` resolves inside content.py.
3. Possibly wipe `workspace/run<N>/assets/` if `--no-cache` passed.
4. Dynamically import content module; confirm it exposes `build(story, kit, STYLES)`.
5. Set matplotlib backend to `Agg` before calling build (idempotent).
6. Construct `story = []`; call `mod.build(story, notes_kit, notes_kit.STYLES)`.
7. Call `notes_kit.generate_notes_pdf(out_path, title, subtitle, lambda s: s.extend(story), theme)`.
8. Catch all exceptions → `errors[]`.
9. Page count via `pypdf`.
10. If `--preview`: rasterize first 2 pages with `pdf2image` at 120 DPI.
11. Collect every file in `workspace/run<N>/assets/` → `assets[]`.
12. Emit `BuildResponse` JSON on stdout. Diagnostics → `stderr` only.

---

## 9. Robustness / Self-Healing Rules (v2 — graphics additions)

v1 rules retained (helpers coerce types, `add_code` auto-escapes, unknown callout kind
falls back to `info`, missing image placeholder, etc.). v2 adds:

### Charts

- `add_bar_chart` with empty `categories` or empty `values` → renders an empty-axis chart
  with a "no data" caption; logs warning, never raises.
- `add_pie_chart` with negative values → coerces `max(0, v)` and logs warning.
- Mismatched `categories` / `values` length → truncates to shorter pair, logs warning.
- Multi-series with ragged `values` lists → pads with zeros, logs warning.

### Diagrams

- `add_flowchart` (and friends) when `dot` binary missing → insert a red-bordered
  Paragraph: `[graphviz 'dot' not installed; flowchart skipped]` + log warning.
- Empty `edges` list → same placeholder + warning.
- Self-loops / duplicate edges → graphviz handles silently.

### Math

- `add_math` with invalid LaTeX → matplotlib mathtext raises; kit catches → output the
  expression verbatim in monospace + warning (so the agent sees the bad formula in the
  PDF instead of a build failure).
- Empty `expr` → insert a blank thin spacer + warning.

### Code highlighting

- Unknown language to pygments → falls back to plain escaped code (v1 path) + warning.
- Pygments import fails (unlikely) → same fallback path. No build failure.

### Asset cache

- If `assets/<hash>.png` exists for a helper-call hash, skip re-rendering — embed the
  cached file. This means: re-running `build` for the same `content.py` is fast, and
  partial edits don't re-render unchanged graphics.
- If a cached PNG is corrupted (PIL fails to open), delete it and re-render once.

---

## 10. Dependency Budget (v2)

Pin in `pyproject.toml` (additions marked NEW):

| Package      | Version | Purpose                                   | Status |
|--------------|---------|-------------------------------------------|--------|
| reportlab    | >=4.0   | PDF rendering engine                       | v1     |
| pdfplumber   | >=0.10  | validate page parsing                       | v1     |
| pypdf        | >=4.0   | page counting                              | v1     |
| pdf2image    | >=1.17  | PNG previews (system poppler-utils)         | v1     |
| matplotlib   | >=3.8   | charts + math rendering (mathtext)          | v1     |
| pydantic     | >=2.0   | schemas                                    | v1     |
| **pygments** | >=2.18  | **syntax highlighting in `add_code`**     | NEW    |
| **graphviz** | >=0.20  | **diagrams (needs system `dot` binary)**    | NEW    |
| **numpy**    | >=1.26  | **histogram/box plot data massaging**       | NEW    |
| **pillow**   | >=10.0  | **image inspection in validate, asset cache check** | NEW |
| seaborn      | —       | NOT added (matplotlib suffices; lean deps) | rejected |

System deps (documented in README + SKILL.md):

- `poppler-utils` — for `pdf2image` previews (v1).
- `graphviz` — for `dot` binary used by `add_flowchart` / `add_tree` / `add_state_diagram`
  / `add_block_diagram` / `add_graph` (v2). Without it these helpers degrade to a
  placeholder; charts and math still work.

Install commands committed in README:

```bash
# In repo:
uv sync
# System (Debian/Ubuntu):
sudo apt install poppler-utils graphviz
```

---

## 11. Graphics Pipeline Detail

### 11.1 Universal rendering recipe (every graphics helper)

Every graphics helper follows the same 6-step recipe:

1. **Validate/coerce inputs** (per §9) — never raise.
2. **Hash the call** using `hashlib.sha1(repr((helper_name, args_tuple))).hexdigest()[:8]`
   to produce a stable asset filename like `bar_chart_a1b2c3d4.png`.
3. **Check cache**: if `<run_dir>/assets/<filename>` exists AND is openable by PIL, embed
   that file via `notes_kit.add_image(story, path, width=width, caption=caption)` and return.
4. **Render**: call the matplotlib / graphviz code to produce a PNG at the run's
   `assets/<filename>` path.
5. **Embed**: same `add_image` call as step 3.
6. **Warnings**: append any degradations to a module-level `_warnings` list the runner
   collects at the end of the build. (Done via a tiny `warn(msg)` helper that routes to
   this list.)

Why this matters: steps 3+5 mean rebuilding for an unrelated text edit doesn't re-render
charts (matplotlib is the slowest part). For a 6-chart PDF, this turns a 4-second rebuild
into a 0.4-second one.

### 11.2 Asset cache layout

```
workspace/run1/
└── assets/
    ├── bar_chart_a1b2c3d4.png
    ├── line_chart_9e8f7a6b.png
    ├── tree_00112233.png
    ├── math_44556677.png
    └── highlighted_code_aa11bb22.png   # if we cache highlight renders (we do not by default)
```

Cleanup rules:
- `runner.py build --no-cache` wipes `assets/` before rendering.
- Orphan PNGs (in `assets/` but not in the current build's hash set) are left alone (cheap
  and safe; the agent might re-add a removed call). Optional cleanup command can be added
  later (`runner.py gc --run workspace/run1`).

### 11.3 Theming injection

`graphics_kit.py` does NOT redefine colors. It imports from `notes_kit`:

```python
from notes_kit import (INK, HEADING, ACCENT, ACCENT_2, MUTED, BORDER, PAGE_SIZE, MARGIN,
                       CHART_PALETTE, CHART_BG, CHART_GRID_COLOR, ...)
```

Before calling matplotlib, graphics_kit sets a module-level matplotlib style override:

```python
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [BODY_FONT],
    "text.color": str(INK),
    "axes.labelcolor": str(INK),
    "axes.edgecolor": str(BORDER),
    "xtick.color": str(MUTED),
    "ytick.color": str(MUTED),
    "axes.grid": True,
    "grid.color": str(CHART_GRID_COLOR),
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
})
```

This is set once on import (inside a `_setup_matplotlib()` called at module top). Every
chart inherits the brand look automatically.

---

## 12. Graphics Helper Catalog (the v2 API surface)

Each helper below mutates `story` in place, accepts a trailing `caption=` and `width=`,
never raises, and follows the universal rendering recipe in §11.1. Width defaults to
`CHART_DEFAULT_W` (170 mm) for charts, `DIAGRAM_DEFAULT_W` for diagrams, `None` for math
(auto-fit).

### 12.1 Charts (matplotlib — pure python, no system dep)

#### `add_bar_chart(story, title=None, categories=[], series=None, values=None, xlabel=None, ylabel=None, horizontal=False, caption=None, width=None)`

- Series form: `series=[{"label": "A", "values": [...]}, ...]` (multi-series grouped bars)
  OR short-form `values=[1,2,3]` (single series, anonymous).
- `horizontal=True` → barh orientation.
- Bars use `CHART_PALETTE` rotation; agent cannot override colors.
- Hatches reserved for a future "print-friendly" mode — out of scope for v2 actual.

#### `add_line_chart(story, title=None, series=[], xlabel=None, ylabel=None, markers=True, caption=None, width=None)`

- `series=[{"label": "Naive", "xs": [...], "ys": [...]}, ...]`.
- Each series gets a distinct color from `CHART_PALETTE` and a distinct linestyle
  (`-`, `--`, `:`, `-.`) cycling if more series than palette entries.
- `markers=True` (default) places a small dot at each data point.

#### `add_pie_chart(story, title=None, labels=[], values=[], donut=False, caption=None, width=None)`

- `donut=True` → donut chart (inner radius 0.6).
- Negative values coerced to 0 with warning.
- Legend placed to the right if labeled slices > 4; below otherwise.

#### `add_scatter(story, title=None, xs=[], ys=[], labels=None, xlabel=None, ylabel=None, trendline=False, caption=None, width=None)`

- Single-series by default. If `labels` given, each point gets that label and points are
  colored by `CHART_PALETTE` based on label group.
- `trendline=True` overdraws a least-squares fit (numpy `polyfit` degree 1) in a dashed
  muted-gray line.

#### `add_histogram(story, title=None, data=[], bins=20, xlabel=None, ylabel=None, caption=None, width=None)`

- `data` is a flat list of numbers. `bins` is an int or a list of edges.

#### `add_box_plot(story, title=None, groups=None, values=None, xlabel=None, ylabel=None, caption=None, width=None)`

- `groups=["A","B","C"]`, `values=[[...],[...],[...]]` (one list of values per group).
- Boxes use `CHART_PALETTE` fills at 0.6 alpha.

#### `add_area_chart(story, title=None, series=[], xlabel=None, ylabel=None, stacked=True, caption=None, width=None)`

- `series=[{"label": "A", "xs": [...], "ys": [...]}, ...]`. xs must match across series.
- `stacked=True` → stacked areas (default).

#### `add_heatmap(story, title=None, matrix=None, row_labels=None, col_labels=None, cmap="Blues", annotate=True, caption=None, width=None)`

- `matrix` is a 2D list of numbers. Rows × cols.
- `cmap` defaults to `"Blues"` (locked unless future theming adds more options).
- Annotates each cell with its value (2-decimal round).

### 12.2 Diagrams (graphviz — requires system `dot` binary; graceful fallback)

All diagram helpers accept the same shape:
```python
edges = [(src, dst, optional_label), ...]
node_labels = {"src": "label", "dst": "label"}   # optional; defaults to node id
caption = "..."
```

The kit builds the DOT source internally (using the `graphviz` Python package), sets
graph/edge defaults for the brand palette, runs `dot -Tpng`, embeds the result. The agent
never writes DOT directly.

#### `add_flowchart(story, edges, node_labels=None, rankdir="TB", caption=None, width=None)`

- `rankdir="TB"` (top-bottom) or `"LR"` (left-right). Default TB.
- Nodes are rounded boxes (`shape=box`, `style="rounded,filled"`, fill = light, edge = ACCENT).
- Decision nodes detected when their `node_labels` value ends with `"?"` → diamond shape.

#### `add_tree(story, edges, node_labels=None, caption=None, width=None)`

- Strict top-down tree layout. Uses `rankdir="TB"`.
- Children of same parent laid out left-to-right automatically by `dot`.
- Root auto-detected (a node with no incoming edge). If multiple roots → forest in one
  diagram (still valid graphviz).

#### `add_state_diagram(story, edges, node_labels=None, initial=None, accepting=None, caption=None, width=None)`

- `initial=[id]` (or string "id") gets a leading arrow drawn (`->` from an invisible source).
- `accepting=["q2", "q3"]` → those nodes get `shape=doublecircle`.
- Other nodes default `shape=circle`.
- Edges default `shape=vee` arrowheads.

#### `add_block_diagram(story, edges, node_labels=None, rankdir="LR", caption=None, width=None)`

- Architecture-style: rectangles with thick BORDER stroke, sans-serif labels.
- `rankdir="LR"` default (left-to-right pipeline look).
- Edge labels supported for "sub-step" annotations.

#### `add_graph(story, edges, node_labels=None, layout="neato", caption=None, width=None)`

- Generic undirected graph. `layout="neato"`|`dot"`|`"circo"`|`"fdp"`.
- Nodes are circles, edges undirected.
- Good for "toy graph" illustrations in DS&A notes.

#### `add_call_graph(story, edges, node_labels=None, caption=None, width=None)`

- Directed graph (`layout=dot`, `rankdir=TB`), `shape=box`.
- For visualizing function call relationships (not strictly a tree).

#### `add_venn(story, sets=None, labels=None, caption=None, width=None)`

- TWO- or THREE-circle Venn only. `sets={"A": set(), "B": set(), "C": set()}` (2 or 3 keys).
- Counts each region (A only, A∩B, etc.) and labels inside each region.
- Rendered with matplotlib (NOT graphviz) since graphviz can't do Venn natively. Uses
  three translucent circles with `ACCENT`/`ACCENT_2`/`emerald` fills at 0.35 alpha.

#### `add_timeline(story, events=None, caption=None, width=None)`

- `events=[{"label": "2020 — paper X", "year": 2020}, {"label": "2022 — improvement Y", "year": 2022}, ...]`.
- Horizontal axis min..max = min(year)..max(year) + 10% padding.
- Each event is a tick mark on the axis with a small label above (alternates above/below
  axis for readability).
- Rendered with matplotlib (not graphviz).

### 12.3 Math (matplotlib mathtext — pure python)

#### `add_math(story, expr, display=True, width=None, caption=None)`

- `expr` is a raw LaTeX-ish string (matplotlib mathtext subset — NOT full LaTeX). No `\begin{...}`.
- `display=True` → larger font (24pt), centered. `display=False` → inline-sized (12pt).
- Rendered to a high-DPI transparent PNG (300 DPI). Width auto-fit unless `width=` given.
- For inline math inside a paragraph, the agent SHOULD prefer `<sub>`/`<super>` tags;
  `add_math` is for display-size standalone formulas.

Supported by matplotlib mathtext: fractions `\frac`, subscripts `x_i`, superscripts `x^2`,
sqrt `\sqrt`, sums `\sum`, integrals `\int`, greek `\alpha \beta \pi`, matrices via `\begin{matrix}...\end{matrix}` (limited), arrows `\rightarrow`. Multi-line formulas via `\\` —
supported inside mathtext.

### 12.4 Code highlighting (pygments)

#### `add_code(story, code, language=None, highlight=False, line_numbers=False, caption=None)`

- v1 behavior kept when `highlight=False` (default) — dark monospace block, no colors.
- `highlight=True` → routes through `highlighting.highlight_to_reportlab(code, language)`
  which produces a colored `Paragraph` (or table-of-paragraphs if `line_numbers=True`)
  using monokai token colors on the existing `CODE_BG` background.
- `line_numbers=True` places a left-side gutter with line numbers in muted color
  (`MUTED`), the code body to the right.
- Unknown language → falls back to plain `highlight=False` style + warning.
- The agent never touches pygments style names directly; the kit locks to **monokai**.

---

## 13. Validation extensions for graphics (reiteration, expanded)

| Kind                  | Severity | How to detect                                                 |
|-----------------------|----------|----------------------------------------------------------------|
| `image_oversized`     | error    | `pdfplumber` `page.images[i].bbox` — if bbox right > frame right + 1pt |
| `low_contrast_image`  | warn     | Open the asset with PIL, compute `(R+G+B)/3 < 128` fraction; flag if <5% AND image area >30% of page |
| `orphan_caption`      | warn     | Caption Paragraph whose top Y is >4pt below the previous flowable's bottom Y on same page |

These run after the build, on the produced `notes.pdf`, in `runner.py validate`. They do
NOT fail the build; they only populate `issues[]`.

---

## 14. Deliverables Checklist (v2)

- [x] `plan.md`         — this file (v2 with graphics).
- [ ] `func.md`         — extend with new graphics snippets (§21 charts, §22 diagrams, §23 math, §24 highlighted code).
- [ ] `notes_kit.py`    — add `from graphics_kit import *` + `from highlighting import *` re-exports; extend `__all__`.
- [ ] `graphics_kit.py` — NEW. Charts + diagrams + math helpers per §12.
- [ ] `highlighting.py` — NEW. Pygments-to-ReportLab formatter + `highlight_to_reportlab` + `add_code` upgrade hook.
- [ ] `runner.py`       — add `assets[]` to BuildResponse; collect `assets/` dir; add 3 new validators; add `--no-cache` flag.
- [ ] `schemas/generate_spec.py` — add `assets: list[str]` to `BuildResponse`.
- [ ] `SKILL.md`        — update description + add graphics-helper table.
- [ ] `README.md`       — document `graphviz` system dep, show chart + diagram examples.
- [ ] `pyproject.toml`  — add `pygments`, `graphviz`, `numpy`, `pillow` deps.
- [ ] End-to-end test: build a "DP notes with recursion tree + bar chart + math formula + highlighted code" PDF; validate; inspect previews.

---

## 15. Future Extensions (v3 notes, not for this round)

- Pygments style menu (optionally expose style names like `monokai`, `tango`, `friendly-dark`).
- Auto-TOC with page numbers (v1 idea §10).
- Two-column reference layout via `BaseDocTemplate` + `Frame` (v1).
- Seaborn-themed statistical plots (rejected in v2 to keep deps lean; revisit if requested).
- Sequence diagrams (mermaid-style) — would need a pure-python renderer (`plantuml` is
  Java-based). Defer until a clean dep emerges.
- Live "edit and refresh" preview server — out of scope; agent loop already handles iteration.
- High-DPI vector graphics embedded as SVG (ReportLab supports limited SVG upload) —
  re-render charts as `.svg` for crispr large-format printing.
- Sparkline helper for embedding mini-charts inside table cells (would need a per-cell
  Image flowable; doable but adds API surface).

---

## 16. Decision Log (v2)

- 2026-07-07: split graphics into `graphics_kit.py` rather than bloating `notes_kit.py`
  (keeps the kit source browsable; agent still imports a single name).
- 2026-07-07: ship the v2 chart palette as a fixed rotation — no agent color arguments.
  Rationale: weak models can't pick harmonious colors; the palette is designed to look
  good in any order.
- 2026-07-07: diagram dep is `graphviz` python wrapper, requiring system `dot` binary.
  Charts avoid this dep entirely (matplotlib has bundled fonts). Decision: rather than
  implement a fallback matplotlib-based diagram renderer (which always looks worse),
  degrade to a placeholder telling the agent to fall back to text. Better a clean "not
  available" than a bad diagram.
- 2026-07-07: asset cache keyed by stable hash of `(helper_name, args_repr)`. Decision:
  `repr()` chosen over JSON because dicts in args don't yield stable JSON keys but do
  produce stable `repr()` for simple types. For agent-supplied data this should be fine;
  mutable defaults (`memo={}` in code, etc.) are NOT passed by the agent so this isn't a
  concern.
- 2026-07-07: pygments style locked to monokai. Single style = consistent look. Future
  v3 can expose a style kwarg once we have a small curated allowlist.
- 2026-07-07: `add_code(highlight=False)` by default. v1 callers see no behavior change;
  agents opt in via `highlight=True`.
- 2026-07-07: `add_math` uses matplotlib mathtext (no external latex install). Acceptance:
  limited but covers 95% of study-note needs (sums, integrals, fractions, matrices).
- 2026-07-07: line numbers on code are opt-in; defaults preserve v1's compact look.
- 2026-07-07: Venn and timeline use matplotlib (not graphviz) since their geometry isn't
  graph-layout-shaped. Two renderers (`dot`, matplotlib) in `graphics_kit.py` is fine.
- 2026-07-07: `numpy` added — needed for `polyfit` trendline, `histogram` bin computations,
  and reliable box-plot stats. Pillow + skia not needed beyond PIL image inspection.

---

## End of plan.md (v2)