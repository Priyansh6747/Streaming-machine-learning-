# CLAUDE.md

MkDocs (Material) documentation site for a research project on streaming machine
learning. Content lives in `docs/`, split into `Phase0:Research/` and
`Phase1:Experimentation/`. Config: `mkdocs.yml`. Deployed to GitHub Pages by
`.github/workflows/ci.yml` on every push to `main`.

## Agent tooling: use `.agentInstructions/`

All agent-facing material (prompts, skills, taste files, house style) belongs in
`.agentInstructions/`. **Do not create new top-level dot-folders** for tooling
(`.commandcode`, `.cursor`, `.aider`, etc.) and do not add files under `.claude/`
beyond its existing settings. See `.agentInstructions/README.md`.

## Commands

- `uv run playwright install chromium` — one-time setup for the PDF export feature
- `uv run mkdocs serve` — preview locally (set `MKDOCS_EXPORTER_PDF=false` to skip PDF generation if the browser is unavailable)
- `uv run mkdocs build --strict` — build; fails on broken links

Every page has a **Download as PDF** button (mkdocs-exporter plugin). PDFs are
rendered at build time in headless Chromium, so `mkdocs build`/`mkdocs serve`
require the Playwright browser unless `MKDOCS_EXPORTER_PDF=false` is set.
