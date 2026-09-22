# struct.md — what every file in this repo does

**Scope:** everything in the repo **except `docs/`** (the documentation site has its
own 38-page structure and is deliberately excluded here). Covers the state as of
2026-09-22.

**Repo:** `github.com/Priyansh6747/Streaming-machine-learning-` — a research +
development project on streaming machine learning, concept drift, and the "Delay
Tax". Published site: `https://priyansh6747.github.io/Streaming-machine-learning-/`.

---

## 0. Where the actual work lives

Four things matter; everything else is tooling around them:

| Path | Role |
|---|---|
| `experiments/` | **The experiments** — runners, data artifacts, figure generators |
| `docs/` | **The evidence** — write-ups, literature review, results (excluded from this doc) |
| `.agentInstructions/` | Agent tooling — skills, taste, house rules (not part of the project) |
| root config | Packaging, CI, docs build (`pyproject.toml`, `mkdocs.yml`, `uv.lock`, `.github/`) |

`main.py` is a stub and does nothing. The project is not an application — it is an
experiment suite plus a documentation site.

---

## 1. Root files

| File | What it does |
|---|---|
| `README.md` | The project **vision page** — plain language, no code (deliberate). Status table, the mechanism argument, headline evidence, the roadmap. First thing an examiner reads. |
| `CLAUDE.md` | Agent-facing instructions for this repo: points agents at `.agentInstructions/`, forbids new top-level dot-folders, and lists the mkdocs commands. |
| `main.py` | **Stub.** `print("Hello from major!")`. No real entry point — an experiment is run by invoking its script directly. |
| `pyproject.toml` | `uv` project definition. Deps: `deep-river`, `river` (pinned transitively `<0.26`), `torch`, `scikit-learn`, `matplotlib`, `pytest` (a workaround — deep-river imports it at package-import time but omits it from its own deps), `mkdocs-material`, `mkdocs-exporter`. Also restricts `[tool.uv] environments` to Linux + Apple-Silicon arm64 to avoid the Intel-Mac torch wheel blocker. |
| `uv.lock` | Locked, fully-resolved dependency graph (~122 KB). Makes the environment reproducible. |
| `.python-version` | `3.12` — pinned because deep-river's `torch <2.6` has no cp313 wheels. |
| `mkdocs.yml` | The docs site config: Material theme, MathJax + Mermaid, admonitions/tables/footnotes, the `exporter` plugin (per-page PDFs + optional aggregated `documentation.pdf` via headless Chromium), and the full `nav` tree. **A page under `docs/` is invisible on the site unless it is in this `nav`.** |
| `.gitignore` | Three entries only: `site/`, `.venv/`, `__pycache__/`. |
| `notes.pdf` | **26-page generated PDF** — "The Delay Tax — Literature Review". Produced 2026-08-25 by the `pdf-notes` skill (`workspace/run1`). Covers scope, the gap verdict, lemmas 1–4, scoop risk, closest prior work, baseline pick, go/no-go, and every reviewed paper with its numbers. |

> **Not a source of truth:** `notes.pdf` is a *generated artifact*. Its content is
> derived from the literature-review notes in `docs/Phase0:Research/`. Regenerating it
> means re-running the skill, not editing the PDF.

---

## 2. `experiments/` — the experiment suite

One directory per experiment, plus shared `scripts/` and `figures/`. Every claim in
the docs traces to a `results.json` in here.

```
experiments/
├── exp01/results.json       # Exp-01 artifact
├── exp02/results.json       # Exp-02 artifact
├── figures/                 # shared figure source
│   └── make_thesis_schematic.py
└── scripts/                 # all runners + figure generators
    ├── exp01_deepriver_vs_frozen_sea.py
    ├── exp02_seed_sweep_sea.py
    ├── make_exp01_summary_chart.py
    ├── make_results_figures.py
    └── make_sept9_report_graphs.py
```

### 2a. Data artifacts

| File | What it holds |
|---|---|
| `exp01/results.json` | Exp-01 (seed 42): the **full per-sample** rolling-Kappa series plus all drift-window and full-run metrics. The per-sample series is what makes the rolling-Kappa curve re-renderable. |
| `exp02/results.json` | Exp-02 (11 seeds): per-seed rows (recovery, Δκ, damage, full-run acc/κ), all aggregate distributions (mean/SD/median/min/max), the acceptance scorecard, the Exp-01 cross-check, and the scenario/model spec. **Does not retain the per-sample series** — which is why the metric fix in ML.md §11 needs a rerun. |

### 2b. Runners

| File | What it does |
|---|---|
| `scripts/exp01_deepriver_vs_frozen_sea.py` | Exp-01: builds the SEA stream with one abrupt drift at τ=30,000, runs DeepRiver rolling vs frozen RF prequentially over 48k post-warm-up samples, records per-sample metrics, writes `exp01/results.json`. Flags: `--device cuda\|cpu`. |
| `scripts/exp02_seed_sweep_sea.py` | Exp-02: the same experiment as a loop over 11 seeds. Writes `results.json` **and** re-renders the 9-figure suite. Flags: `--require-cuda`, `--limit` (dev slice), `--figure-only` (regenerate all figures from the committed artifact with **no rerun**), `--figures-dir`. |
| `scripts/make_exp01_summary_chart.py` | Figure generator: the Exp-01 recovery-time vs damage-area summary chart. |
| `scripts/make_results_figures.py` | Figure generator for `docs/Reports/Results.md` — process-flow and headline-scoreboard figures. |
| `scripts/make_sept9_report_graphs.py` | Figure generator for the Sept-9 milestone report — roadmap and expected-vs-measured charts. |
| `figures/make_thesis_schematic.py` | Draws the *illustrative* three-curve recovery schematic (streaming / scheduled / frozen) used in `thesis.md` and the README. Note it lives in `figures/`, not `scripts/` — the only figure script not in `scripts/`. |

> **Run an experiment:** `uv run experiments/scripts/exp02_seed_sweep_sea.py --require-cuda`
> **Re-render figures only:** `uv run experiments/scripts/exp02_seed_sweep_sea.py --figure-only`

> **Known structural weakness:** `exp01` and `exp02` **duplicate the metric code**
> (`drift_metrics`) by hand — the Exp-02 docstring says the definitions are "copied
> verbatim from the Exp-01 script so Exp-02 is a sweep of one experiment, not a new
> one". There is no shared, config-driven harness yet; that is the Chapter-1 exit
> criterion and the main missing development artifact.

---

## 3. `.agentInstructions/` — agent tooling (not project content)

Policy (from `README.md`): this is the **single home** for anything instructing an AI
agent — prompts, skills, taste. No new top-level dot-folders; keep it in version
control.

| Path | What it does |
|---|---|
| `README.md` | The policy itself, plus one hard project rule: **every experiment must produce a plot or data artifact — no experiment may end with only prose.** |
| `taste/taste.md` | Index file; points at the category file. |
| `taste/taste/taste.md` | Captured preferences about **research and documentation style** — verify citations before marking them verified, flag risks loudly, dense/numbers-first deliverables, figures in docs, structured markdown over prose, etc. |

### 3a. `skills/pdf-notes/` — a complete PDF-generation skill

A self-contained sub-project (its own `.venv`, `pyproject.toml`, `uv.lock`,
`.python-version`) that turns "notes-as-code" into polished PDFs. It produced
`notes.pdf`.

| File | What it does |
|---|---|
| `SKILL.md` | Skill metadata + trigger description + quick start + the helper-category table. |
| `README.md` | Install and usage: system deps (`poppler-utils`, `graphviz`), the `init`/`build`/`validate` workflow. |
| `plan.md` | 40 KB architecture doc — v2 design with graphics: helper catalogue, rendering recipe, asset cache, validation heuristics, design tokens, decision log. |
| `plan_detailed_mode.md` | The chapter-stitched generation mode: instead of one giant `content.py`, plan a manifest and write one small `.py` chapter per turn. |
| `func.md` | 27 KB snippet library the agent copies verbatim, editing only the data strings. |
| `notes_kit.py` | Core rendering: design tokens, styles, text helpers, `NotesCanvas`, `generate_notes_pdf`, and re-exports from the two modules below. |
| `graphics_kit.py` | Charts (matplotlib), diagrams (graphviz), math (mathtext), Venn, timeline. The largest module (~58 KB). |
| `highlighting.py` | Pygments → ReportLab-XML syntax highlighting (monokai) for `add_code(highlight=True)`. |
| `file_tree.py` | Pure-vector file-tree diagram Flowable — draws drive/folder/file nodes with connectors and arrowheads, no image deps. |
| `runner.py` | CLI: `init` / `build` / `validate`. Emits JSON contracts on stdout, diagnostics on stderr. |
| `schemas/generate_spec.py` | Pydantic request/response models for the JSON contract. |
| `schemas/__init__.py` | Empty package marker. |
| `tests/` | Pytest suite: `test_detailed_mode.py`, `test_file_tree.py`, `test_highlighting.py`. |
| `main.py` | Stub — `print("Hello from pdf-skill!")`. Legacy, unused by the skill. |
| `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore` | The skill's own isolated environment (Python 3.13), independent of the project's. |
| `workspace/run1/` | **The actual generation run that produced `notes.pdf`.** |
| `workspace/run1/book.json` | The manifest: title, subtitle, theme, and the ordered chapter list. |
| `workspace/run1/chapters/ch01…ch16_*.py` | 16 chapter sources — `overview`, `gap`, `lemma1`–`lemma4`, `scoop`, `closest_prior`, `baseline`, `gono`, `papers_ch1`, `ch1b`, `ch2`, `ch3`, `ch3b`, `hybrid`. |
| `workspace/run1/assets/*.png` | Rendered math/chart assets, cache-keyed by hash. |
| `workspace/run1/preview_p1…26.png` | Per-page preview renders (one per page of the 26-page PDF). |

> `workspace/run1/` has no `notes.pdf` in it — the built PDF was copied up to the repo
> root as `notes.pdf`.

---

## 4. `.commandcode/` — legacy duplicate of the taste files

| Path | What it does |
|---|---|
| `taste/taste.md` | Index pointer (identical content to the `.agentInstructions/` one). |
| `taste/taste/taste.md` | Captured preferences about **experiment and execution style** — persist experiment code as real scripts in `experiments/scripts/`, use the GPU, keep script docstrings lean, one narrative per figure, draft commit messages but don't commit, keep CI in sync with build deps, a page in `docs/` is only "done" when it's in the nav. |

> **Inconsistency worth knowing:** `.agentInstructions/README.md` states this folder
> **was renamed** to `.agentInstructions` and rule 1 forbids new top-level dot-folders
> for agent tooling. It nonetheless exists again — recreated by the tooling. The two
> `taste/` trees are **not** mirrors: they hold different captured preferences. Treat
> both as live.

### Other dot-paths

| Path | What it does |
|---|---|
| `.claude/settings.local.json` | Editor-managed permission allowlist (currently one entry: `WebFetch(domain:riverml.xyz)`). |
| `.github/workflows/ci.yml` | Deploy job: on push to `main` **when `docs/**`, `mkdocs.yml`, or the workflow changed**, installs `mkdocs-material` + `mkdocs-exporter`, installs the Playwright Chromium, then `mkdocs gh-deploy --force` with PDF export on. |

> **Consequence for adding pages:** a commit that only touches, say, `experiments/`
> will **not** trigger a redeploy. Only changes under `docs/` or to `mkdocs.yml` do.

---

## 5. Environment and commands

| Command | Effect |
|---|---|
| `uv sync` | Install/refresh the project environment |
| `uv run experiments/scripts/exp0N_*.py` | Run an experiment (GPU if present) |
| `uv run experiments/scripts/exp02_seed_sweep_sea.py --figure-only` | Re-render the Exp-02 figure suite from the committed artifact — no rerun |
| `uv run mkdocs serve` | Local docs preview (needs the Playwright browser for PDFs) |
| `MKDOCS_EXPORTER_PDF=false uv run mkdocs serve` | Fast local preview, no PDF rendering |
| `uv run mkdocs build --strict` | Build; fails on broken links |
| `uv run playwright install chromium` | One-time setup for the PDF export feature |

**Hardware reality:** the DeepRiver loop needs the GPU. 484 samples/s on the RTX 3050
vs 44 samples/s on CPU for the same 48k-sample loop — ~11× — so `--require-cuda` on
Exp-02 exists to fail fast rather than silently take an hour.

---

## 6. Things to know before touching this repo

1. **`docs/` is the published artifact, not a scratch folder.** Anything placed there
   is built into the site and, with the exporter on, into downloadable PDFs.
2. **A page in `docs/` without a `nav` entry is invisible** — it builds and is
   reachable by URL, but nothing links to it. Three pages are currently in that state:
   `papers/dasari-2026-when-to-retrain.md`, `papers/read-2012-batch-vs-instance.md`,
   and `Reports/_milestone-template.md`.
3. **`main.py` does nothing.** The real entry points are the scripts in
   `experiments/scripts/`.
4. **The two runners duplicate the metric code.** Any change to how recovery or damage
   is defined must be made in both, or the comparison between Exp-01 and Exp-02 breaks.
5. **`notes.pdf` is generated, not authored** — it is the output of
   `skills/pdf-notes/workspace/run1`, not a hand-written document.
6. **`.commandcode/` shouldn't exist** per the repo's own policy, but does, and holds
   unique preferences. Don't delete it without moving its content.
7. **Figure scripts live in two places** (`experiments/scripts/` and
   `experiments/figures/`) for no documented reason.
8. **`exp02/results.json` has no per-sample series** — a metric-definition fix there
   requires re-running the sweep, not post-processing.
9. **Do not edit `.agentInstructions/taste/` or `.commandcode/taste/` by hand.**
   They're managed by the learning system.
