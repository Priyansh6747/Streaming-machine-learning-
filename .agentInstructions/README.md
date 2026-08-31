# .agentInstructions

The single home for anything that instructs or configures an AI agent working in
this repo: prompts, skills, taste files, workflow notes, house style.

## Rules for agents

1. **Do not create new top-level dot-folders** for agent tooling — no
   `.commandcode`, no `.cursor`, no `.aider`, no `.windsurf`, no new
   `.<tool>` directory, and do not add files under `.claude/` beyond the
   editor-managed settings already there. Put everything here instead, under a
   descriptive subfolder (`skills/`, `taste/`, …).
2. This folder was previously named `.commandcode`. If you find references to
   that name, update them to `.agentInstructions`.
3. Keep it in version control. Do not gitignore this folder.

## Contents

- `skills/` — packaged task instructions (e.g. `pdf-notes/`)
- `taste/` — house style / taste guidance
