---
name: pdf-notes
description: Use this skill to generate polished, study-ready PDF notes with charts, diagrams, math formulas, syntax-highlighted code, and external image/PDF/SVG import from any source material. Supports bar/line/pie/scatter charts, flowcharts, trees, state machines, Venn diagrams, timelines, LaTeX math, highlighted code blocks, and rasterizing SVG diagrams or PDF pages into the notes. Trigger whenever the user wants clean, structured "notes" as a PDF rather than a raw document dump.
license: Personal use
---

# PDF Notes Generation Skill

Generate visually rich, study-ready PDFs from **notes-as-code** — a sequence of predefined helper calls. The agent never imports `reportlab`, `matplotlib`, `graphviz`, or `pygments` directly.

## Quick Start

```bash
# 1. Scaffold a run folder
uv run python runner.py init --name my_notes

# 2. Edit workspace/run1/content.py (see func.md for snippets)

# 3. Build the PDF
uv run python runner.py build \
    --content workspace/run1/content.py \
    --out workspace/run1/notes.pdf \
    --title "My Topic" \
    --subtitle "Study notes" \
    --preview

# 4. Validate layout
uv run python runner.py validate --pdf workspace/run1/notes.pdf
```

## System Dependencies

```bash
# Required for previews
sudo apt install poppler-utils
# Required for diagrams (flowcharts, trees, state machines)
sudo apt install graphviz
```

## Files

- `notes_kit.py` — all helpers (text + re-exports from graphics_kit/highlighting)
- `graphics_kit.py` — charts, diagrams, math, Venn, timeline
- `highlighting.py` — pygments syntax highlighting for `add_code(highlight=True)`
- `runner.py` — CLI: `init`, `build`, `validate`
- `func.md` — snippet library the agent copies from verbatim
- `plan.md` — full architecture and agent workflow
- `schemas/generate_spec.py` — Pydantic contracts for JSON I/O

## Agent Workflow

1. Plan outline (nested bullets: H1 -> H2 -> H3 -> content blocks).
2. Init a run folder: `uv run python runner.py init --name <topic>`.
3. Copy snippets from `func.md` into `content.py` — only edit string arguments.
4. Build: `uv run python runner.py build --content ... --out ... --title ... --preview`.
5. Parse JSON output. If `success == false`, fix errors and rebuild.
6. Validate: `uv run python runner.py validate --pdf ...`.
7. Fix any `severity: error` issues. Iterate until clean.

## Helper Categories

| Category   | Helpers                                                  | Deps        |
|------------|----------------------------------------------------------|-------------|
| Text       | `add_title_block` `add_heading` `add_paragraph` `add_bullets` `add_numbered_list` `add_code` `add_callout` `add_table` `add_divider` `add_space` `add_page_break` `add_quote` `add_image` `add_keep_together` | reportlab   |
| Charts     | `add_bar_chart` `add_line_chart` `add_pie_chart` `add_donut_chart` `add_scatter` `add_histogram` `add_box_plot` `add_area_chart` `add_heatmap` `add_radar_chart` | matplotlib  |
| Diagrams   | `add_flowchart` `add_tree` `add_state_diagram` `add_block_diagram` `add_graph` `add_call_graph` `add_venn` `add_timeline` | graphviz + matplotlib |
| Math       | `add_math`                                               | matplotlib  |
| Code       | `add_code(highlight=True, line_numbers=True)`            | pygments    |
| Source Import | `add_image_rasterize`                                  | cairosvg / rsvg-convert / inkscape / convert (SVG); pdf2image + poppler (PDF); PIL (bitmaps) |

## Detailed Mode (Chapter-Stitched Generation)

For large notes, use detailed mode to avoid single-turn output limits. Instead of
one giant `content.py`, the agent plans a Table of Contents (manifest), writes
**one small `.py` chapter per turn**, and the runner stitches them into one PDF.

### When to Use Detailed Mode

- Notes expected to be **longer than ~500 lines** of helpers
- Complex multi-section topics (math, code, charts, diagrams in one doc)
- When per-chapter iteration speed matters (fix one chapter, rebuild fast)

### Workflow

```bash
# 1. Scaffold a detailed run folder
uv run python runner.py init --name dp_notes --mode detailed
# -> creates workspace/runN/book.json + chapters/ch01_*.py, ch02_*.py, ch03_*.py

# 2. Edit book.json (TOC/outline) — only id, slug, title, subtitle, summary, file fields
# 3. Write each chapter file (chapters/chXX_<slug>.py) — one per turn
# 4. Build one chapter for testing
uv run python runner.py build \
    --content workspace/runN/book.json \
    --out workspace/runN/notes.pdf \
    --title "Dynamic Programming" \
    --only ch03 --preview
# 5. Build all chapters (stitched)
uv run python runner.py build \
    --content workspace/runN/book.json \
    --out workspace/runN/notes.pdf \
    --preview
# 6. Validate (unchanged)
uv run python runner.py validate --pdf workspace/runN/notes.pdf
```

### Key Commands

| Command | Purpose |
|---------|---------|
| `--mode detailed` | Scaffold book.json + chapters/ tree |
| `--only ch02,ch03` | Render only listed chapter ids |
| `--skip ch04` | Exclude listed chapter ids |
| `--content X.json` | Auto-detected as detailed mode |
| `--content X.py` | Auto-detected as single mode (backward compat) |

### Chapter File Shape (unchanged contract)

Same as `content.py` — `from notes_kit import *` + `def build(story, kit, STYLES)`. Optional hooks:

- `suppress_chapter_title = False` — skip auto title block
- `chapter_title_override = "Custom"` — override manifest title
- `starts_on_new_page = True` — page break before this chapter
- `show_chapter_divider = False` — thin rule after auto title

### Response Fields

Detailed mode adds these fields to BuildResponse:
- `chapters_rendered: int` — how many chapters built successfully
- `chapters_failed: list[str]` — chapter ids that failed (callout placed in PDF)
- `chapter_pages: dict[str, dict]` — maps each chapter id to its page range:
  ```json
  {"ch01": {"start": 1, "end": 3, "pages": [1, 2, 3]}, "ch02": {"start": 4, "end": 5, "pages": [4, 5]}}
  ```

### Preview PNGs

- Preview PNGs are **always generated** for all pages (not just first 2)
- Old `preview_p*.png` files are **automatically cleaned** at the start of each build
- Use `chapter_pages` to know which preview PNG corresponds to which chapter

---

## External Image Import (SVG / PDF Pages)

Use `add_image_rasterize` to embed source material that isn't already a bitmap.
It rasterizes the source to a cached PNG in `assets/`, then embeds via `add_image`.

| Source type | What happens |
|-------------|-------------|
| **SVG** | Rasterized via cairosvg (preferred), rsvg-convert, inkscape, or convert. |
| **PDF** | Rasterized via pdf2image (already a project dependency; needs poppler). |
| **Bitmap** (PNG/JPG/GIF/BMP/WEBP) | Passed straight through to `add_image` — no conversion. |

```python
# SVG diagram copied from source
kit.add_image_rasterize(story, "assets/architecture.svg", caption="System architecture")

# A page from an external PDF
kit.add_image_rasterize(story, "source/chapter5.pdf", page=3, caption="Diagram from Ch.5", dpi=200)
```

**Key args:**

| Arg | Default | Effect |
|-----|---------|--------|
| `page` | `1` | 1-indexed page number (PDF only) |
| `dpi` | `200` | Rasterization resolution. 200 for screen, 300+ for print |
| `bg` | `"white"` | Background fill. Pass `None` to preserve transparency |
| `width` / `caption` | `None` | Same as `add_image` — auto-fit to frame if `width` is `None` |

**Caching:** Rasterized PNGs are keyed by `(path, mtime, page, dpi, bg)` — re-running
build after editing only text skips unchanged rasterizations. `--no-cache` forces
full re-render.

**Graceful degradation:** If no SVG rasterizer is found, a clear error message is
recorded as a warning and a red placeholder paragraph is placed in the PDF — the build
never crashes.

## Key Rules

- **Never import reportlab/matplotlib/graphviz/pygments** in content.py.
- **Never use Unicode sub/super glyphs** (₀ ²). Use `<sub>`/`<super>` tags or `add_math`.
- **Always use helpers** (`kit.add_*`). Never call `story.append(...)` directly.
- **Keep code lines under ~80 chars** to avoid overflow validation errors.
- **Don't end with `add_page_break`** — creates empty trailing page.
- **Graphics helpers never raise** — they degrade gracefully with warnings.
