# func.md — Snippet Library for the Agent

> Copy snippets from this file **verbatim**; only edit the *string arguments* and the `level=`
> / `kind=` / `language=` flags. Do not invent new helpers. Do not import reportlab. Do not
> touch styles directly.
>
> Each snippet has three parts: **Usage**, **Args**, **Snippet**. The kit is `notes_kit`,
> imported via `from notes_kit import *`. In `content.py` the agent calls them as
> `kit.add_*(...)` where `kit` is passed by the runner.

---

## 0. Mandatory header of every content.py

Every `content.py` MUST start like this. Nothing else at top.

```python
from notes_kit import *

def build(story, kit, STYLES):
    # ---- agent writes its notes-as-code below this line ----
    pass
```

`STYLES` is exposed for advanced use only — don't reach for it; prefer the `add_*` helpers.

---

## 1. Title block

`add_title_block(story, title, subtitle=None)`

```python
kit.add_title_block(story, "Dynamic Programming",
                    "Core patterns, recognition, and complexity")
```
- `title`    str. Shown as large bold title.
- `subtitle` optional str. Muted line under the title. Pass `None` to skip.

---

## 2. Headings (H1 / H2 / H3)

`add_heading(story, text, level=1)`

```python
kit.add_heading(story, "Dynamic Programming", level=1)
kit.add_heading(story, "Recognizing DP Problems", level=2)
kit.add_heading(story, "Bottom-Up Tabulation", level=3)
```
- `level` 1 (large), 2 (blue accent), 3 (smaller). Default `1`.
- H1 gets a thin rule under it automatically.

---

## 3. Body paragraph (supports inline tags)

`add_paragraph(story, text)`

```python
kit.add_paragraph(story,
    "DP solves problems by combining solutions to overlapping subproblems, "
    "storing results to avoid recomputation.<br/><br/>"
    "Time complexity: O(n<super>2</super>) naive, O(n log n) with a Fenwick tree.")
```
- Allowed inline tags: `<b> <i> <u> <sub> <super> <font color="#hex"> <br/>`.
- Use `<br/>` for line breaks; sequences of "<br/><br/>" make a blank line.

---

## 4. Bullet list

`add_bullets(story, items, bullet_char="•")`

```python
kit.add_bullets(story, [
    "<b>Optimal substructure</b> — optimal solution built from optimal sub-solutions",
    "<b>Overlapping subproblems</b> — the same subproblem recurs many times",
])
```
- Each item is a paragraph string (inline tags OK).
- Override bullet char with `bullet_char="*"` etc. Default `•`.

---

## 5. Numbered list

`add_numbered_list(story, items)`

```python
kit.add_numbered_list(story, [
    "Define the <b>state</b>: dp[i] = min cost to reach step i.",
    "Write the <b>transition</b>: dp[i] = min(dp[i-1], dp[i-2]) + cost[i].",
    "Pick a base case: dp[0] = 0.",
])
```

---

## 6. Code block (dark surface, monospace)

`add_code(story, code, language=None, highlight=False, line_numbers=False, caption=None)`

```python
# Plain (v1 default)
kit.add_code(story, """def fib(n, memo={}):
    if n <= 1:
        return n
    if n not in memo:
        memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
    return memo[n]""", language="python")

# Syntax-highlighted (pygments monokai)
kit.add_code(story, "x = 42\nprint(x)", language="python", highlight=True)

# With line numbers
kit.add_code(story, "for i in range(10):\n    print(i)",
             language="python", highlight=True, line_numbers=True)
```
- Pass raw code as a triple-quoted string. The kit **auto-escapes** `& < >`.
- `language` optional. If given, prints a small uppercase label above the block.
- `highlight=True` enables syntax coloring (requires pygments). Default `False`.
- `line_numbers=True` adds a line-number gutter (only with `highlight=True`).
- **Watch line length** — keep lines under ~80 chars to avoid `_code_overflow` validation errors.

---

## 7. Callout box

`add_callout(story, text, kind="info", title=None)`

Four kinds: `info`, `tip`, `warning`, `important`. Kit fills in default title if `title=None`.

```python
kit.add_callout(story,
    "Always define your <b>state</b> and <b>transition</b> before coding.",
    kind="tip")

kit.add_callout(story,
    "Watch for integer overflow when N > 10<super>9</super>.",
    kind="warning", title="Edge case")

kit.add_callout(story,
    "Never mutate the input array while iterating it.",
    kind="important")
```

---

## 8. Table (header + zebra rows)

`add_table(story, headers, rows, col_widths=None)`

```python
kit.add_table(story,
    headers=["Approach", "Time", "Space"],
    rows=[
        ["Naive recursion", "O(2<super>n</super>)", "O(n)"],
        ["Memoization",     "O(n)",            "O(n)"],
        ["Tabulation",      "O(n)",            "O(n) or O(1)"],
    ])
```
- `col_widths` optional list of `pts`. If omitted, columns share width equally.
- Sums of widths MUST equal `PAGE_SIZE[0] - 2*MARGIN` (~170*mm on A4) to avoid overflow:
```python
kit.add_table(story, headers=["A", "B"], rows=[["1","2"]],
              col_widths=[60*mm, 110*mm])
```

---

## 9. Divider

`add_divider(story)`

```python
kit.add_divider(story)
```
- Thin horizontal hairline; use to separate major sections.

---

## 10. Spacer

`add_space(story, height=8)`

```python
kit.add_space(story, height=12)
```
- `height` in points. Default 8.

---

## 11. Page break

`add_page_break(story)`

```python
kit.add_page_break(story)
```
- Force next block onto a new page. Avoid putting one at the very end (creates an empty page).

---

## 12. Blockquote

`add_quote(story, text, attribution=None)`

```python
kit.add_quote(story,
    "Premature optimization is the root of all evil.",
    attribution="Donald Knuth")
```

---

## 13. Image + caption

`add_image(story, path, width=None, caption=None)`

```python
kit.add_image(story, "workspace/run1/diagram.png",
              width=120*mm, caption="Recursion tree for fib(5).")
```
- `width` optional, in points or `mm`. Aspect ratio preserved.
- `caption` optional muted text below.
- If the file is missing the kit emits a red "[missing image]" placeholder instead of crashing.

---

## 14. Keep block together (anti-orphan)

`add_keep_together(story, *builders)`

Each builder is a lambda that takes a list `s` and calls `add_*` on it.

```python
kit.add_keep_together(story,
    lambda s: kit.add_heading(s, "Binary Search", level=2),
    lambda s: kit.add_paragraph(s, "Runs in O(log n)."),
    lambda s: kit.add_code(s, "lo, hi = 0, n\nwhile lo < hi:\n    mid = (lo+hi)//2",
                           language="python"))
```
- Use whenever you suspect a heading + first block could get split across a page.
- Also great for code blocks that shouldn't detach from their preceding label/caption.

---

## 15. Bar chart

`add_bar_chart(story, title=None, categories=[], series=None, values=None, xlabel=None, ylabel=None, horizontal=False, caption=None, width=None)`

```python
# Single series
kit.add_bar_chart(story,
    title="Time complexity",
    categories=["Naive", "Memo", "Tabulation"],
    values=[16, 4, 2],
    ylabel="ms",
    caption="Figure 1 — lower is better")

# Multi-series
kit.add_bar_chart(story,
    title="Calls vs n",
    categories=["n=1", "n=2", "n=3", "n=4", "n=5"],
    series=[
        {"label": "Naive", "values": [1, 2, 4, 8, 16]},
        {"label": "Memo",  "values": [1, 2, 3, 4, 5]},
    ],
    ylabel="call count",
    caption="Figure 2 — exponential vs linear")
```
- Single-series: pass `values=[...]`. Multi-series: pass `series=[{"label":..., "values":...}, ...]`.
- `horizontal=True` for horizontal bars.
- Colors are locked to the skill palette (agent cannot override).

---

## 16. Line chart

`add_line_chart(story, title=None, series=[], xlabel=None, ylabel=None, markers=True, caption=None, width=None)`

```python
kit.add_line_chart(story,
    title="Growth comparison",
    series=[
        {"label": "O(n^2)", "xs": [1,2,3,4,5], "ys": [1,4,9,16,25]},
        {"label": "O(n)",   "xs": [1,2,3,4,5], "ys": [1,2,3,4,5]},
    ],
    xlabel="n", ylabel="operations",
    caption="Figure 3 — quadratic vs linear")
```
- `markers=True` (default) places dots at data points.

---

## 17. Pie / Donut chart

`add_pie_chart(story, title=None, labels=[], values=[], donut=False, caption=None, width=None)`

```python
kit.add_pie_chart(story,
    title="Language distribution",
    labels=["Python", "C++", "Other"],
    values=[60, 30, 10],
    caption="Figure 4")

# Donut variant
kit.add_donut_chart(story,
    title="Language distribution",
    labels=["Python", "C++", "Other"],
    values=[60, 30, 10])
```

---

## 18. Scatter plot

`add_scatter(story, title=None, xs=[], ys=[], labels=None, xlabel=None, ylabel=None, trendline=False, caption=None, width=None)`

```python
kit.add_scatter(story,
    title="Runtime vs input size",
    xs=[100, 500, 1000, 5000, 10000],
    ys=[0.1, 2.3, 9.5, 230, 950],
    xlabel="n", ylabel="seconds",
    trendline=True,
    caption="Figure 5")
```
- `trendline=True` adds a least-squares fit line.

---

## 19. Histogram

`add_histogram(story, title=None, data=[], bins=20, xlabel=None, ylabel=None, caption=None, width=None)`

```python
kit.add_histogram(story,
    title="Runtime distribution",
    data=[1.1, 1.3, 1.2, 1.5, 1.4, 1.1, 1.3, 1.2, 1.6, 1.4],
    bins=10,
    xlabel="seconds", ylabel="count",
    caption="Figure 6")
```

---

## 20. Box plot

`add_box_plot(story, title=None, groups=None, values=None, xlabel=None, ylabel=None, caption=None, width=None)`

```python
kit.add_box_plot(story,
    title="Runtime by algorithm",
    groups=["Naive", "Memo", "Tab"],
    values=[[16,18,15,17,16], [4,5,4,3,4], [2,3,2,2,3]],
    ylabel="ms",
    caption="Figure 7")
```

---

## 21. Heatmap

`add_heatmap(story, title=None, matrix=None, row_labels=None, col_labels=None, annotate=True, caption=None, width=None)`

```python
kit.add_heatmap(story,
    title="DP table for LCS",
    matrix=[[0,1,2,3],[1,1,2,3],[2,2,2,3]],
    row_labels=["", "A", "B"],
    col_labels=["", "X", "Y", "Z"],
    caption="Figure 8")
```

---

## 22. Radar chart

`add_radar_chart(story, title=None, categories=[], series=None, values=None, fill=True, alpha=0.25, caption=None, width=None)`

```python
# Single series
kit.add_radar_chart(story,
    title="Skill proficiency",
    categories=["Python", "Algorithms", "Databases", "Networking", "Security"],
    values=[90, 75, 80, 60, 70],
    caption="Figure 9 — self-assessed skills")

# Multi-series comparison
kit.add_radar_chart(story,
    title="Language comparison",
    categories=["Speed", "Safety", "Ecosystem", "Learning curve", "Tooling"],
    series=[
        {"label": "Rust",   "values": [95, 98, 60, 40, 70]},
        {"label": "Python", "values": [40, 50, 95, 90, 90]},
    ],
    caption="Figure 10 — trade-off radar")
```
- Single-series: pass `values=[...]`. Multi-series: pass `series=[{"label":..., "values":...}, ...]`.
- `categories` labels each axis. If omitted, auto-labeled V1, V2, …
- `fill=True` (default) fills the polygon area. `alpha` controls fill transparency.

---

## 23. Flowchart (graphviz)

`add_flowchart(story, edges, node_labels=None, rankdir="TB", caption=None, width=None)`

```python
kit.add_flowchart(story,
    edges=[
        ("start", "check", ""),
        ("check", "base", "n <= 1"),
        ("check", "recurse", "n > 1"),
        ("base", "return", ""),
        ("recurse", "return", ""),
    ],
    node_labels={
        "start": "Input n",
        "check": "Is n <= 1?",
        "base": "return n",
        "recurse": "fib(n-1)+fib(n-2)",
        "return": "Output",
    },
    caption="Figure 9 — Fibonacci flowchart")
```
- `rankdir="TB"` (top-bottom) or `"LR"` (left-right).
- Node labels ending with `?` become diamond shapes (decisions).
- Requires system `dot` binary (`sudo apt install graphviz`).

---

## 24. Tree diagram (graphviz)

`add_tree(story, edges, node_labels=None, caption=None, width=None)`

```python
kit.add_tree(story,
    edges=[
        ("fib(5)", "fib(4)"),
        ("fib(5)", "fib(3)"),
        ("fib(4)", "fib(3)"),
        ("fib(4)", "fib(2)"),
        ("fib(3)", "fib(2)"),
        ("fib(3)", "fib(1)"),
    ],
    caption="Figure 10 — recursion tree for fib(5)")
```

---

## 25. State diagram (graphviz)

`add_state_diagram(story, edges, node_labels=None, initial=None, accepting=None, caption=None, width=None)`

```python
kit.add_state_diagram(story,
    edges=[
        ("q0", "q1", "a"),
        ("q1", "q2", "b"),
        ("q2", "q0", "a"),
    ],
    node_labels={"q0": "Idle", "q1": "Processing", "q2": "Done"},
    initial="q0",
    accepting=["q2"],
    caption="Figure 11")
```

---

## 26. Block diagram (graphviz)

`add_block_diagram(story, edges, node_labels=None, rankdir="LR", caption=None, width=None)`

```python
kit.add_block_diagram(story,
    edges=[("input", "parse"), ("parse", "process"), ("process", "output")],
    node_labels={"input": "Raw Data", "parse": "Parser", "process": "Engine", "output": "Result"},
    caption="Figure 12 — pipeline architecture")
```

---

## 27. Graph (undirected, graphviz)

`add_graph(story, edges, node_labels=None, layout="neato", caption=None, width=None)`

```python
kit.add_graph(story,
    edges=[("A", "B"), ("B", "C"), ("C", "A"), ("A", "D")],
    caption="Figure 13 — simple graph")
```

---

## 28. Call graph (graphviz)

`add_call_graph(story, edges, node_labels=None, caption=None, width=None)`

```python
kit.add_call_graph(story,
    edges=[("main", "solve"), ("solve", "dfs"), ("dfs", "dfs")],
    node_labels={"main": "main()", "solve": "solve(n)", "dfs": "dfs(u)"},
    caption="Figure 14 — call graph")
```

---

## 29. Venn diagram (matplotlib)

`add_venn(story, sets=None, labels=None, caption=None, width=None)`

```python
kit.add_venn(story,
    sets={"DP": {1,2,3,4,5}, "Greedy": {3,4,5,6,7}},
    labels={"DP": "Dynamic Prog.", "Greedy": "Greedy"},
    caption="Figure 15 — overlap = problems solvable both ways")
```
- Supports 2 or 3 sets only.

---

## 30. Timeline (matplotlib)

`add_timeline(story, events=None, caption=None, width=None)`

```python
kit.add_timeline(story,
    events=[
        {"year": 1957, "label": "Bellman coins DP"},
        {"year": 1979, "label": "Khachiyan ellipsoid"},
        {"year": 1984, "label": "Karmarkar interior point"},
    ],
    caption="Figure 16 — optimization milestones")
```

---

## 31. Math formula (matplotlib mathtext)

`add_math(story, expr, display=True, width=None, caption=None)`

```python
# Display-size formula
kit.add_math(story,
    r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}",
    caption="Figure 17 — arithmetic series")

# Integral
kit.add_math(story,
    r"\int_0^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}")

# Matrix
kit.add_math(story,
    r"\begin{matrix} 1 & 2 \\ 3 & 4 \end{matrix}")
```
- `display=True` (default) renders large and centered. `display=False` for inline-sized.
- Uses matplotlib mathtext (NOT full LaTeX). Supports: `\frac`, `\sum`, `\int`, `\sqrt`,
  subscripts `_i`, superscripts `^2`, greek letters, `\rightarrow`, matrices via
  `\begin{matrix}...\end{matrix}`.

---

## 32. Highlighted code (pygments)

`add_code(story, code, language=None, highlight=True, line_numbers=True, caption=None)`

```python
kit.add_code(story,
    "def fib(n, memo={}):\n"
    "    if n <= 1: return n\n"
    "    if n not in memo:\n"
    "        memo[n] = fib(n-1) + fib(n-2)\n"
    "    return memo[n]",
    language="python", highlight=True, line_numbers=True,
    caption="Figure 18 — memoized fibonacci")
```
- `highlight=True` enables monokai syntax coloring. Default `False` (plain monospace).
- `line_numbers=True` adds a line-number gutter. Default `False`.
- Falls back to plain code if pygments is unavailable or language is unknown.

---

## 33. File/folder tree diagram

`add_file_tree(story, spec, caption=None)`

```python
kit.add_file_tree(story, {
    "label": "C:\\", "node_type": "drive", "children": [
        {"label": "Program Files", "node_type": "folder", "children": [
            {"label": "...", "node_type": "ellipsis"},
        ]},
        {"label": "Windows", "node_type": "folder", "children": [
            {"label": "...", "node_type": "ellipsis"},
        ]},
        {"label": "...", "node_type": "ellipsis"},
    ],
})

# D:\ drive with nested folders and files
kit.add_file_tree(story, {
    "label": "D:\\", "node_type": "drive", "children": [
        {"label": "Java", "node_type": "folder", "children": [
            {"label": "beginner", "node_type": "folder", "children": [
                {"label": "Hello.java", "node_type": "file"},
                {"label": "Hello.class", "node_type": "file"},
            ]},
            {"label": "Advanced", "node_type": "folder", "children": [
                {"label": "Pong.java", "node_type": "file"},
                {"label": "Pong.class", "node_type": "file"},
            ]},
        ]},
        {"label": "...", "node_type": "ellipsis"},
    ],
})
```
- `spec` nested dict with `node_type` (drive | folder | file | ellipsis), `label`, optional `children` list.
- `caption` optional muted text below the diagram.
- Tree automatically paginates — splits across pages at safe branch boundaries (never splits a node from its children mid-branch).
- Renders drive icon (stacked bars), folder icon (tab + body rect), file icon (rect with fold lines) per node_type.
- `"ellipsis"` nodes render as literal "..." text.
- `label` is required for all types except `"ellipsis"`.

---

## 34. Full content.py template with graphics (copy this and edit)

```python
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_title_block(story, "TOPIC TITLE", "one-line subtitle")

    kit.add_heading(story, "Section 1", level=1)
    kit.add_paragraph(story, "Intro paragraph.")

    kit.add_heading(story, "Diagram", level=2)
    kit.add_tree(story,
        edges=[("root", "child1"), ("root", "child2")],
        caption="Figure 1")

    kit.add_heading(story, "Chart", level=2)
    kit.add_bar_chart(story,
        title="Comparison",
        categories=["A", "B", "C"],
        values=[10, 20, 15],
        caption="Figure 2")

    kit.add_heading(story, "Formula", level=2)
    kit.add_math(story, r"E = mc^2", caption="Figure 3")

    kit.add_heading(story, "Code", level=2)
    kit.add_code(story, "print('hello')", language="python",
                 highlight=True, line_numbers=True)

    kit.add_callout(story, "Key insight here.", kind="tip")
    kit.add_divider(story)
    kit.add_paragraph(story, "End of notes.")
```

---

## 35. CLI invocation cheatsheet

Always run from the repo root with the same CWD.

```bash
# Scaffold a fresh run folder with content.py template
uv run python runner.py init --name dp_notes
# -> creates workspace/run1/content.py

# Build PDF + first-2-page PNG previews
uv run python runner.py build --content workspace/run1/content.py \
                       --out    workspace/run1/notes.pdf \
                       --title  "Dynamic Programming" \
                       --subtitle "Patterns and complexity" \
                       --preview
# -> prints BuildResponse JSON on stdout (includes assets[])

# Force re-render all graphics (ignore cache)
uv run python runner.py build --content workspace/run1/content.py \
                       --out    workspace/run1/notes.pdf \
                       --title  "DP" --no-cache

# Validate rendered PDF (returns issues[])
uv run python runner.py validate --pdf workspace/run1/notes.pdf
# -> prints ValidateResponse JSON on stdout
```

---

## 36. Common-repair recipe (what to edit for each diagnostic)

| Diagnostic from JSON       | Action in content.py                                       |
|----------------------------|------------------------------------------------------------|
| `code_overflow`            | Shorten lines in the named `add_code` block.              |
| `orphan_heading`           | Wrap that heading + next block in `add_keep_together`.    |
| `text_touches_margin`      | Pass `col_widths` on the offending `add_table`.           |
| `image_oversized`          | Pass a smaller `width=` on the offending helper.          |
| `overflow`                 | Shrink the longest paragraph or insert `add_page_break`.  |
| `empty_page`               | Delete the trailing `add_page_break`.                     |
| Warning: `dot missing`     | Replace diagram helper with `add_numbered_list` of edges. |
| Warning: `chart failed`    | Check data types; rerun with `--no-cache`.                |
| ImportError: 'add_…'       | Add missing `kit.` prefix.                                |
| NameError: STYLES          | Add `from notes_kit import *` at top (snippet §0).        |

---

## 37. Inline-tag quick math cheats

```python
"<b>bold</b>"
"<i>italic</i>"
"<u>underline</u>"
"x<super>2</super>"
"H<sub>2</sub>O"
"red <font color=\"#DC2626\">alert</font> word"
"line1<br/>line2"
```
Never paste raw Unicode super/subscripts (², ₀); they render as black boxes.
For display-size formulas, use `add_math()` instead.

---

## 38. Anti-patterns (DO NOT do these)

- ❌ Importing `reportlab`/`matplotlib`/`graphviz`/`pygments` in `content.py`.
- ❌ Referencing `STYLES["H1"]` directly to make a heading — use `kit.add_heading`.
- ❌ Defining your own `ParagraphStyle` to "custom" a paragraph — use a justified `add_paragraph`.
- ❌ Putting numeric strings like `O(2^n)` with the caret inside `add_code` *only* — for body
  text it must be `O(2<super>n</super>)`.
- ❌ Calling `story.append(...)` directly — always go through an `add_*` helper.
- ❌ Inserting `add_page_break` as the very last call.
- ❌ Hand-rolling bullet characters (`"- item"`) instead of using `add_bullets`.
- ❌ Passing `color=`, `linewidth=`, `figsize=` to chart helpers — the kit owns styling.
- ❌ Passing `set` objects to `add_venn` that aren't plain Python sets.
- ❌ **APPENDING to content.py when editing** — you must REPLACE the relevant section.
  `content.py` defines the ENTIRE document. Each `build` call creates a fresh PDF from
  scratch. If you add new lines without removing old ones, you get duplicates.
- ❌ **Flowcharts with >8 nodes using `rankdir="TB"`** — they'll be too tall for the page.
  Use `rankdir="LR"` for left-to-right layout, or simplify to <8 nodes, or split across
  multiple flowcharts.

---

## End of func.md


---

## 39. External image import (SVG / PDF pages)

`add_image_rasterize(story, path, width=None, caption=None, page=1, dpi=200, bg="white")`

Embeds an external source by rasterizing it to a cached PNG first.
SVG → cairosvg / rsvg-convert / inkscape / convert. PDF pages → pdf2image.
Bitmaps (PNG/JPG/GIF/BMP/WEBP) pass straight through to `add_image` — no conversion.

```python
# SVG diagram copied from source
kit.add_image_rasterize(story, "assets/architecture.svg", caption="System architecture")

# A page from an external PDF
kit.add_image_rasterize(story, "source/chapter5.pdf", page=3, caption="Diagram from Ch.5", dpi=200)

# Preserve transparency (default bg="white" adds white fill)
kit.add_image_rasterize(story, "assets/logo.svg", bg=None, width=100*mm)

# Pass bitmap through unchanged (no rasterization)
kit.add_image_rasterize(story, "assets/photo.jpg", caption="Photo")
```

| Arg | Default | Effect |
|-----|---------|--------|
| `page` | `1` | 1-indexed page number (PDF only) |
| `dpi` | `200` | Rasterization resolution. 200 for screen, 300+ for print |
| `bg` | `"white"` | Background fill. Pass `None` to preserve transparency |
| `width` / `caption` | `None` | Same as `add_image` — auto-fit to frame if `width` is `None` |

Caching: rasterized PNGs keyed by `(path, mtime, page, dpi, bg)` — only re-rendered when
source changes. `--no-cache` forces full re-render.

Degrades gracefully: missing rasterizer → warning + red placeholder paragraph; build never crashes.


---

## 40. Detailed notes mode

### book.json schema (TOC manifest)

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
    }
  ]
}
```

- `id` — stable short key, used by `--only` / `--skip` flags.
- `slug` — human-friendly, also used in filenames.
- `title`/`subtitle` — auto-rendered as title block at start of each chapter (unless suppressed).
- `file` — path relative to manifest directory.
- `summary` — free-text planning scaffold, not rendered.

### Chapter template (chapters/chXX_<slug>.py)

```python
from notes_kit import *

# Optional module-level hooks:
# suppress_chapter_title = False   # skip auto title block
# chapter_title_override = None    # override manifest title
# starts_on_new_page = True        # page break before (ignored for ch01)
# show_chapter_divider = False     # thin rule after auto title

def build(story, kit, STYLES):
    """Chapter: Intro — Motivation, when to reach for DP."""
    kit.add_heading(story, "Section 1", level=1)
    kit.add_paragraph(story, "Body text goes here.")
```

### Three-turn build sequence example

```bash
# Turn 1: Init
uv run python runner.py init --name dp_notes --mode detailed
# Edit book.json with full TOC plan

# Turn 2: Write chapters one at a time (each in its own turn)
# ... write chapters/ch01_intro.py
# ... write chapters/ch02_core.py
# ... write chapters/ch03_advanced.py

# Turn 3: Build all
uv run python runner.py build \
    --content workspace/run1/book.json \
    --out workspace/run1/notes.pdf \
    --title "Dynamic Programming" --preview

# Incremental test (while writing):
uv run python runner.py build \
    --content workspace/run1/book.json \
    --out workspace/run1/notes.pdf \
    --only ch03 --preview
```

### Error recovery

If a chapter fails, the PDF still builds — the failed chapter produces a visible
callout in the document. The agent fixes only that chapter file and rebuilds.

```bash
# Check which chapters failed
# -> chapters_failed: ["ch02"]

# Fix chapters/ch02_core.py, then rebuild
uv run python runner.py build \
    --content workspace/run1/book.json \
    --out workspace/run1/notes.pdf --preview
```

### Output fields (additional)

| Field | Type | Description |
|-------|------|-------------|
| `chapters_rendered` | `int` | Chapters built successfully |
| `chapters_failed` | `list[str]` | Chapter ids that failed |
| `chapter_pages` | `dict[str, dict]` | Maps chapter id → `{"start": N, "end": M, "pages": [N,...,M]}` |
| `warnings` | `list[str]` | Includes "Unreferenced chapter file:" warnings |

### Preview PNGs

- Preview PNGs are **always generated** for all pages (not just first 2)
- Old `preview_p*.png` files are **automatically cleaned** at start of each build
- Use `chapter_pages` to know which preview PNG corresponds to which chapter
