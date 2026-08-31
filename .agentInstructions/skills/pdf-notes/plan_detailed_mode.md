# Plan — Detailed Notes Mode (Chapter-Stitched Generation)

## Problem

The current single-`content.py` flow forces the LLM to emit the **entire** document
in one tool call. For large notes this:

- Exceeds single-turn output limits → truncated / silently dropped chapters.
- Degrades quality → later sections get less attention as context fills.
- Makes fixes expensive → one typo requires rewriting a 2000-line file.
- Risks losing work → a build error near the end wastes the whole long output.

We need a "detailed notes mode" where the agent first plans a Table of Contents
(TOC), then emits **one small `.py` chapter file at a time**, and the runner
**stitches** them into one PDF. No quality loss vs. single-file mode — same
helpers, same styling, same validation.

---

## Design Principles (non-negotiables)

1. **No quality regression.** Both modes share the exact same `notes_kit`
   helpers, `NotesCanvas`, themes, asset cache, validators. Detailed mode is
   purely an *authoring* ergonomics layer; rendering is unchanged.
2. **Backward compatible.** Existing single-file `content.py` workflow and
   `runner.py build --content X.py` must keep working unmodified.
3. **Notes-as-code preserved.** Chapters are still `.py` files with a
   `build(story, kit, STYLES)` function — **not** Markdown. Markdown would
   lose chart/diagram/math/code-highlighting fidelity.
4. **Failure isolation.** One bad chapter must not kill the whole build.
   Chapter modules are imported defensively; errors are surfaced per chapter but
   the rest still render. **Automatic retry** (up to 3 attempts) with story
   rollback and module re-import on each failure.
5. **One PDF out.** All chapters always concatenate into a single document
   with continuous page numbering.

---

## Architecture: What Changes

### Filesystem layout (detailed run)

```
workspace/runN/
├── book.json                 # TOC / manifest — the "plan" the agent writes first
├── chapters/
│   ├── ch01_intro.py
│   ├── ch02_concepts.py
│   ├── ch03_math.py
│   └── ch04_summary.py
├── notes.pdf                 # produced artifact (same as today)
├── preview_p1.png            # auto-generated for ALL pages (not just first 2)
├── preview_p2.png            # old previews auto-cleaned at build start
├── preview_pN.png
└── assets/                   # shared asset cache across all chapters
    ├── math_a1b2c3d4.png    # kit-generated cache files (<helper>_<8hex>.png)
    ├── rasterize_e5f6a7b8.png
    └── page_09.png          # user-placed source images (preserved by --no-cache)
```

### `book.json` — the manifest (the real "table of contents")

```jsonc
{
  "title": "Dynamic Programming",
  "subtitle": "From recurrence to code",
  "theme": "light",
  "chapters": [
    {
      "id": "ch01",
      "slug": "intro",
      "title": "Introduction",
      "subtitle": "Why DP exists",
      "file": "chapters/ch01_intro.py",
      "summary": "Motivation, when to reach for DP, brute-force example."
    },
    {
      "id": "ch02",
      "slug": "concepts",
      "title": "Core Mechanics",
      "subtitle": "States & transitions",
      "file": "chapters/ch02_concepts.py",
      "summary": "Overlapping subproblems, optimal substructure, memoization vs tabulation."
    }
  ]
}
```

- `id` — stable short key, used by `--only` / `--skip` flags.
- `slug` — human-friendly, also used in filenames.
- `title`/`subtitle` — auto-rendered as an `add_title_block` at the start of
  each chapter *unless* the chapter module sets `suppress_chapter_title = True`
  or `chapter_title = None`.
- `file` — path relative to the manifest directory.
- `summary` — free-text the agent writes during TOC planning; **not rendered**.
  Used only as a planning scaffold / LLM self-reminder.

### Chapter file shape (unchanged contract)

Each `chapters/chXX_*.py` looks exactly like today's `content.py`:

```python
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_heading(story, "Motivation", level=2)
    kit.add_paragraph(story, "...")
    kit.add_bar_chart(story, title="...", categories=["A","B"], values=[1,2])
    # ...
```

Optional module-level hooks (all opt-in, defaults sensible):

| Hook | Type | Effect |
|------|------|--------|
| `suppress_chapter_title` | `bool` | Skip auto title block for this chapter |
| `chapter_title_override` | `str` | Use this string instead of manifest `title` |
| `starts_on_new_page` | `bool` (default `True`) | If True, runner inserts `add_page_break` before chapter unless it's the first |
| `show_chapter_divider` | `bool` (default `False`) | If True, a thin `add_divider` is drawn after the auto title block |

This keeps each chapter *small and self-contained* — the LLM emits one per
turn, never touching the others.

---

## Runner Changes

### `init` — new flag `--mode`

```bash
uv run python runner.py init --name dp_notes --mode detailed
```

- `--mode {single,detailed}` (default `single` → existing behavior).
- In `detailed` mode:
  - Scaffold writes `book.json` (minimal stub with one example chapter) and
    `chapters/ch01_intro.py` (commented template).
  - Output JSON: `{success, run_dir, manifest_path, chapters_dir, ...}`.
- `single` mode unchanged (writes `content.py`).

### `build` — accept manifest OR content file

Detection rule (zero new flags required):

> If `--content` points at a `.json` file → detailed (manifest) mode.
> If it points at a `.py` file → single mode (existing behavior).

Detailed-mode build pipeline (`cmd_build_detailed`):

1. Load & validate `book.json` via Pydantic (`BookManifest`).
2. Resolve `theme` from manifest unless `--theme` overrides; same for
   `--title`/`--subtitle` overrides (CLI beats manifest).
3. Setup `assets/` dir + `graphics_kit.set_run_dir()` (shared cache — unchanged).
4. Set matplotlib `Agg` backend (unchanged).
5. Initialize `story = []`.
6. `os.chdir(run_dir)` so relative paths like `assets/page_09.png` resolve.
7. For each `chapter` in `manifest.chapters` (in declared order):
   a. Verify `chapter.file` exists; if not, record `error` per-chapter and
      **continue** (failure isolation).
   b. Defensive dynamic import via `importlib.util.spec_from_file_location`.
      On import exception → `errors[]` entry tagged with chapter id, continue.
   c. Confirm `mod.build` callable; else per-chapter error, continue.
   d. If not first chapter and `starts_on_new_page` (default) →
      `kit.add_page_break(story)`.
   e. If `suppress_chapter_title` is False →
      `kit.add_title_block(story, chapter_title, chapter_subtitle)`.
   f. If `show_chapter_divider` → `kit.add_divider(story)`.
   g. Call `mod.build(story, notes_kit, notes_kit.STYLES)` with **retry**:
      - Up to 3 attempts per chapter.
      - On failure: roll back partial story (`del story[story_before:]`),
        re-import module (so agent file edits take effect), retry.
      - On final failure: append visible `add_callout(...)` warning + mark failed.
8. After all chapters: same `generate_notes_pdf()` call as today (no change).
9. Same `graphics_kit.get_graphics_warnings()`, page count, preview, assets
   collection, same `BuildResponse` JSON shape (add fields:
   `chapters_rendered: int`, `chapters_failed: list[str]`,
   `chapter_pages: dict[str, dict]`).
10. Restore original CWD in `finally` block.

**New `--only` / `--skip` flags** (optional, useful for incremental dev):

```
--only ch02,ch03      Render only listed chapter ids (still a valid PDF).
--skip ch04           Exclude listed chapter ids.
```

Useful while iterating on a single chapter — the agent re-runs build in
seconds for one chapter instead of the whole book. Default: render all.

### Preview PNG Management

- Old `preview_p*.png` files are **automatically cleaned** at the start of each build.
- Preview PNGs are **always generated for ALL pages** (not just first 2).
- The `--preview` flag is deprecated (kept for backward compat; previews are
  always generated).

### Chapter → Page Mapping

After PDF generation, the runner uses `pdfplumber` to search for chapter titles
in the PDF text. Each chapter's page range is determined by:

1. Finding the page where the chapter title appears.
2. Finding the page where the next chapter title appears (or end of document).
3. Mapping `chapter_id → {start, end, pages}`.

This mapping is returned as `chapter_pages` in the build response:

```json
{
  "chapter_pages": {
    "ch01": {"start": 1, "end": 3, "pages": [1, 2, 3]},
    "ch02": {"start": 4, "end": 5, "pages": [4, 5]}
  }
}
```

### `validate`

Unchanged. It scans the final PDF, enum issues by page → still correct because
all chapters stitch into one PDF with continuous pages.

### `assets` cache

Shared across chapters (`run_dir/assets/`). Already keyed by `(helper, args)`
hash, so identical chart in two chapters reuses one PNG. No change needed.

**Smart `--no-cache` wipe:** Only kit-generated cache files (`<helper>_<8hex>.png`)
are deleted. User-placed source images (e.g., `page_09.png`) are preserved.

---

## New Helpers

### `add_image_rasterize` — External Source Import (SVG / PDF Pages)

Embeds external source material by rasterizing it to a cached PNG first.

| Source type | What happens |
|-------------|-------------|
| **SVG** | Rasterized via cairosvg (preferred), rsvg-convert, inkscape, or convert. |
| **PDF** | Rasterized via pdf2image (needs poppler). **Auto-cropped** to content area. |
| **Bitmap** (PNG/JPG/GIF/BMP/WEBP) | Passed straight through to `add_image` — no conversion. |

```python
kit.add_image_rasterize(story, "assets/architecture.svg", caption="System architecture")
kit.add_image_rasterize(story, "source/chapter5.pdf", page=3, caption="Diagram from Ch.5", dpi=200)
```

Key args: `page` (1-indexed PDF page), `dpi` (200 default), `bg` ("white" default, None for transparency),
`width`, `caption`.

### Auto-Crop (`_auto_crop_to_content`)

All rasterized PDF pages and SVGs are **auto-cropped** to the content bounding box:

- PIL/numpy utility that detects the bounding box of non-background content.
- Thresholds at luminance 245 (configurable) to find content pixels.
- Adds 2% padding around content.
- Skips crop if content fills >90% of the page (full-page content).
- Falls back gracefully if numpy/PIL unavailable.

### Unicode Math Normalization (`_normalize_unicode_math`)

All text helpers (`add_paragraph`, `add_heading`, `add_title_block`, `add_bullets`,
`add_numbered_list`, `add_quote`, `add_callout`, image captions) auto-convert
Unicode subscript/superscript glyphs to ReportLab `<sub>`/`<super>` tags:

- `₀₁₂₃₄₅₆₇₈₉ₐₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ₊₋₌₍₎` → `<sub>...</sub>`
- `⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᶜᵈᵉ...ᶻ` → `<super>...</super>`

Prevents black squares (■) from Base14 font missing-glyph fallback.

### Path Resolution (`_resolve_image_path`)

`add_image` and `add_image_rasterize` resolve relative paths against
`graphics_kit.get_run_dir()` as fallback when CWD-based lookup fails.
Ensures `assets/page_09.png` always resolves correctly regardless of CWD state.

### Math Rendering Improvements

- Display fontsize: 12pt (1.14x body of 10.5pt, matches LaTeX `\displaystyle`).
- Inline fontsize: 10.5pt (1.0x body).
- Default width capped: display ≤ 60% frame width, inline ≤ 40%.
- `pad_inches=0` (tighter bounding box, less whitespace).
- Caption fontSize: 10pt (was 8.5pt).

### Spacing Improvements

- Image spacers: captioned 8pt → 4pt, uncaptioned 8pt → 3pt.
- Caption spaceBefore: 2pt → 4pt (tighter visual attachment).

---

## Schemas (in `schemas/generate_spec.py`)

```python
class ChapterSpec(BaseModel):
    id: str
    slug: str
    title: str | None = None
    subtitle: str | None = None
    file: str
    summary: str | None = None

class BookManifest(BaseModel):
    title: str
    subtitle: str | None = None
    theme: str = "light"
    chapters: list[ChapterSpec]

class BookBuildResponse(BuildResponse):
    chapters_rendered: int = 0
    chapters_failed: list[str] = []
    chapter_pages: dict[str, dict] = {}

class InitDetailedResponse(BaseModel):
    success: bool
    run_dir: str
    manifest_path: str
    chapters_dir: str
    chapter_files: list[str]
```

`BookBuildResponse` extends `BuildResponse`, so the agent contract stays a
superset — no breakage.

---

## New Agent Workflow (Detailed Mode)

```
1. PLAN OUTLINE (TOC)
   Agent reasons about the topic, emits book.json to workspace/runN/book.json.
   Each ChapterSpec gets: id, slug, title, subtitle, summary, file path.
   ── Output is small JSON, fits in one turn easily.

2. SCAFFOLD
   uv run python runner.py init --name <topic> --mode detailed
   (Agent either writes book.json itself or edits the scaffolded stub.)

3. WRITE CHAPTERS (one per turn)
   For each chapter in manifest:
     agent writes chapters/chXX_<slug>.py with build(story, kit, STYLES).
     Each chapter is small (≈ one section of a section), emittable in one turn.
   The agent may interleave with: BUILD CHAPTER to test:
     uv run python runner.py build --content workspace/runN/book.json \
         --only ch03 --preview
   Validates the chapter compiles in isolation.

4. FINAL BUILD
   uv run python runner.py build \
       --content workspace/runN/book.json \
       --out workspace/runN/notes.pdf --preview
   └─ Stitches all chapters in declared order → one PDF.

5. VALIDATE (unchanged)
   uv run python runner.py validate --pdf workspace/runN/notes.pdf

6. FIX LOOP (per chapter)
   If a chapter raised → agent edits only that chapter file → rebuild.
   No need to regenerate anything else.
```

Key advantages of writing one chapter per turn:

- **Output stays within limits** — each chapter file is small.
- **Quality preserved** — full attention budget per chapter, feedback between
  chapters improves the next one.
- **Early validation** — `--only` lets the agent compile-test a chapter before
  committing to the next.
- **Recovery** — a broken chapter doesn't invalidate the rest; the agent
  rewrites just that one file.
- **Resume-able** — the agent can stop mid-book and pick up by checking which
  `chapters/*.py` files already exist vs. the manifest.

---

## Backward Compatibility Matrix

| Scenario                              | Behavior                                            |
|---------------------------------------|----------------------------------------------------|
| `init` (no `--mode`)                  | Single mode, writes `content.py` (unchanged)        |
| `init --mode single`                  | Same as above                                       |
| `init --mode detailed`                | Writes `book.json` + `chapters/ch01_*.py` stub      |
| `build --content X.py`                | Single mode (unchanged detection)                   |
| `build --content X.json`              | Detailed mode (new path)                            |
| `validate --pdf X.pdf`                | Unchanged                                           |
| Existing `content.py` files           | Continue to build identically                       |
| `--no-cache`                          | Only wipes kit cache; user images preserved         |
| `--preview`                           | Deprecated (previews always generated now)          |

No existing test or workflow regresses.

---

## Implementation Steps (ordered, each independently testable)

1. **Schemas** — add `ChapterSpec`, `BookManifest`, `BookBuildResponse`,
   `InitDetailedResponse` to `schemas/generate_spec.py`. (Pure data, no risk.)
2. **`cmd_init`** — add `--mode` arg. Branch on mode; in detailed mode write
   `book.json` from a `BOOK_TEMPLATE` constant + commented `ch01_intro.py`
   template under `chapters/`.
3. **`cmd_build_detailed`** — new function. Implement detection in `main()`:
   if `args.content` endswith `.json` → call `cmd_build_detailed`, else
   `cmd_build` (existing). Add `--only` / `--skip` parsing.
4. **Chapter rendering loop** — per the pipeline above, with defensive
   per-chapter try/except and visible callout on failure.
5. **Retry mechanism** — up to 3 attempts per chapter with story rollback
   and module re-import between attempts.
6. **Auto title block insertion** — honor module-level hooks
   (`suppress_chapter_title`, `chapter_title_override`, `starts_on_new_page`,
   `show_chapter_divider`).
7. **Response enrichment** — extend `BuildResponse` with
   `chapters_rendered`, `chapters_failed`, `chapter_pages`.
8. **Preview management** — auto-clean old previews, generate for all pages,
   chapter→page mapping via pdfplumber.
9. **Templates** — author two doc strings:
   - `BOOK_TEMPLATE` (JSON manifest with 2 example chapters).
   - `CHAPTER_TEMPLATE` (a `content.py`-equivalent skeleton geared for a
     single chapter).
10. **`add_image_rasterize`** — external source import (SVG / PDF pages)
    with auto-crop to content area.
11. **Unicode normalization** — `_normalize_unicode_math()` auto-converter
    for subscript/superscript glyphs across all text helpers.
12. **Path resolution** — `_resolve_image_path()` fallback to `get_run_dir()`.
13. **Math sizing** — 12pt display, 10.5pt inline, width cap, tighter spacing.
14. **Caption styling** — 10pt font, 4pt spaceBefore.
15. **Smart `--no-cache`** — `_wipe_kit_cache()` pattern-based deletion.
16. **SKILL.md update** — add "Detailed Mode" section, "External Image Import"
    section, helper categories table.
17. **`func.md` update** — append §39 "External image import", §40 "Detailed
    notes mode" with book.json schema, chapter template, example build sequence.
18. **Tests** — add `tests/test_detailed_mode.py`:
    - `init --mode detailed` creates the expected tree.
    - `build` against a manifest stitches three stub chapters into one PDF
      with correct page ordering and `chapters_rendered == 3`.
    - Intentional `raise` inside `ch02_build()` → retry succeeds on 2nd attempt.
    - `--only ch01` produces a single-chapter PDF.
    - `--no-cache` preserves user-placed images in `assets/`.
    - Unicode subscripts render as `<sub>` tags, not squares.
    - Existing `content.py` test still passes (no regression).

Each step is a single, small PR-sized change. Steps 1–6 unlock the feature;
7–9 polish; 10–18 are docs, helpers, and tests.

---

## Edge Cases & Mitigations

| Risk | Mitigation |
|------|------------|
| Manifest references missing chapter file | Per-chapter error, build continues, callout shown in PDF |
| Chapter `build()` raises mid-render | Retry up to 3x with story rollback; callout on final failure |
| Asset hash collision between chapters | None — hash includes `(helper_name, args)` only; same args ⇒ identical PNG is *correct* reuse |
| Two chapters want same title | Allowed; runner doesn't dedupe. Agent's responsibility. |
| Agent forgets to bump `book.json` after adding a chapter file | Runner reports `error` listing the unreferenced `.py` files in `chapters/` as warnings |
| User passes `.txt`/`.yaml` by mistake | Determined strictly by `.json` vs `.py` suffix; invalid suffix → existing error path |
| Chapter `--only` set but ids unknown | Emit JSON `error` listing unknown ids vs. known ids; exit non-zero |
| Single-page chapter wrapped in keep-together overflow | Unchanged validator catches it; agent edits just that one chapter |
| `--no-cache` wipes user-placed source images | Smart wipe: only kit cache files (`<helper>_<8hex>.png`) deleted |
| Unicode subscripts render as ■ squares | `_normalize_unicode_math()` auto-converts to `<sub>`/`<super>` tags |
| Image paths relative to run_dir don't resolve | `_resolve_image_path()` falls back to `graphics_kit.get_run_dir()` |
| Rasterized PDF pages include full page margins | `_auto_crop_to_content()` trims to content bounding box |

---

## Why this fully solves the stated problem

- **"fails cause the LLM has to directly write the content"** — Agent now writes
  *one chapter* per turn, not the whole document. Output size per turn is
  bounded by chapter size, not book size.
- **"first create a table of contents dividing stuff in chapters"** — That is
  exactly `book.json`, written first, before any rendering code.
- **"then write separate chapter.md for each"** — Separate
  `chapters/chXX_*.py` files (we keep `.py` rather than `.md` to preserve full
  graphical fidelity; a Markdown path would silently downgrade chart/diagram
  expressiveness — outside the "no quality loss" constraint).
- **"then stitch all together"** — Automatic in `build`; the runner imports
  each chapter module in declared order and feeds one shared `story` into the
  existing `generate_notes_pdf()` pipeline. Page numbering, footers, asset
  cache, validators all continue to work unchanged.

The four-step workflow → **Plan TOC → Write chapters one at a time → Stitch →
Validate** — matches the user's proposed approach exactly, while preserving
every visual/structural guarantee the current skill provides.

---

## Commit History (implementation)

| Commit | Summary |
|--------|---------|
| `9d5907c` | Detailed mode + `add_image_rasterize` + SKILL.md/func.md updates |
| `60ac93b` | `mm`/`inch`/`colors` export, retry mechanism, func.md snippet |
| `03f032c` | Math fontsize fix (24→14pt) + relative path CWD fix |
| `33d92ec` | Auto-manage preview PNGs + chapter→page mapping |
| `505b0d1` | Fix 5 PDF rendering regressions (missing images, Unicode squares, math sizing, spacing, captions) |
| `af42329` | Auto-crop rasterized PDF/SVG images to content area |