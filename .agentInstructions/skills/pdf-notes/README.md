# PDF Skill

PDF notes generation skill for OpenClaw agents — produces polished, study-ready PDFs with charts, diagrams, math, and syntax-highlighted code from notes-as-code.

## Install

```bash
uv sync
```

System dependencies:

```bash
# Required for PDF-to-PNG previews
sudo apt install poppler-utils
# Required for diagrams (flowcharts, trees, state machines)
sudo apt install graphviz
```

## Usage

### Scaffold a run

```bash
uv run python runner.py init --name dp_notes
# -> creates workspace/run1/content.py
```

### Edit content.py

Write notes as a sequence of helper calls. See `func.md` for all available snippets.

```python
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_title_block(story, "Dynamic Programming", "Patterns and complexity")
    kit.add_heading(story, "Overview", level=1)
    kit.add_paragraph(story, "DP solves problems by combining overlapping subproblems.")
    kit.add_bullets(story, [
        "<b>Optimal substructure</b> — built from optimal sub-solutions",
        "<b>Overlapping subproblems</b> — same subproblem recurs",
    ])
    kit.add_code(story, "dp[i] = max(dp[i-1], dp[i-2] + arr[i])", language="python")
    kit.add_callout(story, "Define state + transition before coding.", kind="tip")

    # Charts
    kit.add_bar_chart(story,
        title="Time complexity",
        categories=["Naive", "Memo", "Tabulation"],
        series=[{"label": "Time", "values": [16, 4, 2]}],
        ylabel="ms")

    # Diagrams
    kit.add_tree(story,
        edges=[("fib(5)", "fib(4)"), ("fib(5)", "fib(3)"), ("fib(4)", "fib(3)")],
        caption="Recursion tree")

    # Math
    kit.add_math(story, r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}")

    # Highlighted code
    kit.add_code(story, "def fib(n):\n    return n if n<=1 else fib(n-1)+fib(n-2)",
                 language="python", highlight=True, line_numbers=True)
```

### Build PDF

```bash
uv run python runner.py build \
    --content workspace/run1/content.py \
    --out workspace/run1/notes.pdf \
    --title "Dynamic Programming" \
    --subtitle "Core patterns" \
    --preview
```

JSON output on stdout:
```json
{
  "success": true,
  "pdf_path": "workspace/run1/notes.pdf",
  "pages": 4,
  "previews": ["workspace/run1/preview_p1.png", "workspace/run1/preview_p2.png", "workspace/run1/preview_p3.png", "workspace/run1/preview_p4.png"],
  "assets": ["workspace/run1/assets/bar_chart_a1b2.png"],
  "warnings": [],
  "errors": []
}
```

Options:
- `--no-cache` — wipe `assets/` before building (force re-render all graphics)
- Preview PNGs are always generated for all pages (not just first 2)
- Old `preview_p*.png` files are automatically cleaned at start of each build

### Validate layout

```bash
uv run python runner.py validate --pdf workspace/run1/notes.pdf
```

Returns issues: orphan headings, code overflow, empty pages, oversized images.

## Architecture

- `notes_kit.py` — all rendering code (design tokens, styles, text helpers, re-exports)
- `graphics_kit.py` — charts (matplotlib), diagrams (graphviz), math, Venn, timeline
- `highlighting.py` — pygments syntax highlighting (monokai theme)
- `runner.py` — CLI with `init`, `build`, `validate` subcommands
- `func.md` — snippet library for agents (copy verbatim, edit strings only)
- `plan.md` — full architecture, agent workflow, validation heuristics
- `schemas/generate_spec.py` — Pydantic models for JSON contracts
- `workspace/` — sandbox; one subdir per generation run, assets cached in `assets/`
