#!/usr/bin/env python3
"""runner.py — CLI entry point for the PDF notes skill.

Subcommands:
    init      Scaffold a workspace/run<N>/content.py or book.json template
    build     Compile content.py or book.json -> notes.pdf (+ optional preview PNGs)
    validate  Scan a rendered PDF for layout issues

Usage (from repo root):
    uv run python runner.py init --name dp_notes
    uv run python runner.py init --name dp_notes --mode detailed
    uv run python runner.py build --content workspace/run1/content.py --out workspace/run1/notes.pdf --title "DP" --preview
    uv run python runner.py build --content workspace/run1/book.json --out workspace/run1/notes.pdf --title "DP" --preview
    uv run python runner.py validate --pdf workspace/run1/notes.pdf
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `from notes_kit import *` resolves
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# Precompiled hex charset for kit cache filenames
_HEX8 = set("0123456789abcdef")


def _wipe_kit_cache(assets_dir: Path):
    """Remove only kit-generated cached PNGs from assets/.

    Kit files follow the pattern ``<helper>_<8hex>.png`` (e.g. math_a1b2c3d4.png,
    rasterize_644f2f46.png). User-placed source images (page_09.png, diagram.svg,
    etc.) are preserved so they survive ``--no-cache``.
    """
    if not assets_dir.exists():
        return
    for f in assets_dir.iterdir():
        if f.suffix.lower() != ".png":
            continue
        stem = f.stem
        parts = stem.rsplit("_", 1)
        if (len(parts) == 2 and len(parts[1]) == 8
                and all(c in _HEX8 for c in parts[1])):
            f.unlink()

CONTENT_TEMPLATE = '''\
from notes_kit import *

def build(story, kit, STYLES):
    """Agent-authored notes-as-code. Edit below this line."""
    # kit.add_title_block(story, "TOPIC TITLE", "one-line subtitle")
    # kit.add_heading(story, "Section 1", level=1)
    # kit.add_paragraph(story, "Body text goes here.")
    # kit.add_heading(story, "Sub-section", level=2)
    # kit.add_bullets(story, [
    #     "First key point",
    #     "Second key point with <b>bold</b> term",
    # ])
    # kit.add_code(story, "print('hello')", language="python")
    # kit.add_callout(story, "Define state + transition before coding.", kind="tip")
    # kit.add_table(story,
    #     headers=["Approach", "Time", "Space"],
    #     rows=[["Naive", "O(2<super>n</super>)", "O(n)"]])
    # kit.add_divider(story)
    # --- Graphics (v2) ---
    # kit.add_bar_chart(story, title="Chart", categories=["A","B"], values=[1,2])
    # kit.add_line_chart(story, series=[{"label":"L","xs":[1,2],"ys":[3,4]}])
    # kit.add_tree(story, edges=[("root","child1"),("root","child2")])
    # kit.add_math(story, r"\\\\sum_{i=1}^{n} i = \\\\frac{n(n+1)}{2}")
    # kit.add_code(story, "x = 42", language="python", highlight=True, line_numbers=True)
    pass
'''

BOOK_TEMPLATE = '''{
  "title": "TOPIC TITLE",
  "subtitle": "one-line subtitle",
  "theme": "light",
  "chapters": [
    {
      "id": "ch01",
      "slug": "intro",
      "title": "Introduction",
      "subtitle": "Setting the stage",
      "file": "chapters/ch01_intro.py",
      "summary": "Motivation, background, and roadmap for the notes."
    },
    {
      "id": "ch02",
      "slug": "core",
      "title": "Core Mechanics",
      "subtitle": "The main concepts",
      "file": "chapters/ch02_core.py",
      "summary": "Key definitions, algorithms, and structural breakdown."
    },
    {
      "id": "ch03",
      "slug": "advanced",
      "title": "Advanced Topics",
      "subtitle": "Going deeper",
      "file": "chapters/ch03_advanced.py",
      "summary": "Edge cases, optimizations, and extensions."
    }
  ]
}
'''

CHAPTER_TEMPLATE = '''\
from notes_kit import *

# Optional module-level hooks — remove or set as needed:
# suppress_chapter_title = False   # skip auto title block for this chapter
# chapter_title_override = None    # use this string instead of manifest title
# starts_on_new_page = True        # page break before this chapter (ignored for ch01)
# show_chapter_divider = False     # thin rule after the auto title block

def build(story, kit, STYLES):
    """Chapter: {chapter_title} — {chapter_summary}"""
    kit.add_heading(story, "Section 1", level=1)
    kit.add_paragraph(story, "Body text goes here.")
    kit.add_bullets(story, [
        "First key point",
        "Second key point with <b>bold</b> term",
    ])
    kit.add_code(story, "print('hello')", language="python")
    kit.add_callout(story, "Key insight for this section.", kind="tip")
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _next_run_id(workspace: Path) -> int:
    """Find the next available run ID by scanning workspace/run* dirs."""
    existing = []
    if workspace.exists():
        for entry in workspace.iterdir():
            if entry.is_dir() and entry.name.startswith("run"):
                try:
                    num = int(entry.name.removeprefix("run"))
                    existing.append(num)
                except ValueError:
                    pass
    return max(existing, default=0) + 1


def _emit(obj: dict):
    """Print JSON to stdout (the agent contract)."""
    print(json.dumps(obj, indent=2))


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace):
    workspace = REPO_ROOT / "workspace"
    workspace.mkdir(exist_ok=True)

    run_id = _next_run_id(workspace)
    run_dir = workspace / f"run{run_id}"
    run_dir.mkdir(parents=False, exist_ok=True)

    mode = getattr(args, "mode", "single") or "single"

    if mode == "detailed":
        manifest_path = run_dir / "book.json"
        manifest_path.write_text(BOOK_TEMPLATE)

        chapters_dir = run_dir / "chapters"
        chapters_dir.mkdir(parents=False, exist_ok=True)

        manifest = json.loads(BOOK_TEMPLATE)
        chapter_files: list[str] = []
        for ch in manifest["chapters"]:
            ch_path = run_dir / ch["file"]
            ch_path.parent.mkdir(parents=True, exist_ok=True)
            filled = CHAPTER_TEMPLATE.replace(
                "{chapter_title}", ch.get("title", "Untitled")
            ).replace(
                "{chapter_summary}", ch.get("summary", "No summary")
            )
            ch_path.write_text(filled)
            chapter_files.append(str(ch_path.relative_to(REPO_ROOT)))

        _emit({
            "success": True,
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
            "chapters_dir": str(chapters_dir.relative_to(REPO_ROOT)),
            "chapter_files": chapter_files,
            "message": f"Scaffolded run{run_id} (detailed). Edit book.json and chapters/*.py, then build.",
        })
    else:
        content_path = run_dir / "content.py"
        content_path.write_text(CONTENT_TEMPLATE)

        _emit({
            "success": True,
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "content_path": str(content_path.relative_to(REPO_ROOT)),
            "message": f"Scaffolded run{run_id}. Edit content.py, then run build.",
        })


# ---------------------------------------------------------------------------
# build (single mode)
# ---------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace):
    content_path = Path(args.content).resolve()
    out_path = Path(args.out).resolve()
    title = args.title or "Notes"
    subtitle = args.subtitle
    preview = args.preview
    theme = args.theme or "light"
    no_cache = args.no_cache

    errors: list[str] = []
    warnings: list[str] = []
    previews: list[str] = []
    asset_files: list[str] = []

    # 1. Sanity checks
    if not content_path.exists():
        errors.append(f"Content file not found: {content_path}")
        _emit({"success": False, "pdf_path": None, "pages": 0,
               "previews": [], "assets": [], "warnings": [], "errors": errors})
        return

    if not content_path.suffix == ".py":
        errors.append(f"Content file must be .py, got: {content_path.suffix}")
        _emit({"success": False, "pdf_path": None, "pages": 0,
               "previews": [], "assets": [], "warnings": [], "errors": errors})
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Setup assets dir
    run_dir = out_path.parent
    assets_dir = run_dir / "assets"
    if no_cache:
        _wipe_kit_cache(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Clean old preview PNGs
    for old_preview in run_dir.glob("preview_p*.png"):
        old_preview.unlink()

    # 3. Dynamic import of content module
    try:
        spec = importlib.util.spec_from_file_location("content", str(content_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        errors.append(f"ImportError in content.py: {type(e).__name__}: {e}")
        _emit({"success": False, "pdf_path": None, "pages": 0,
               "previews": [], "assets": [], "warnings": [], "errors": errors})
        return

    if not hasattr(mod, "build") or not callable(mod.build):
        errors.append("content.py must define build(story, kit, STYLES)")
        _emit({"success": False, "pdf_path": None, "pages": 0,
               "previews": [], "assets": [], "warnings": [], "errors": errors})
        return

    # 4. Set matplotlib backend
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass

    # 5. Run the build
    import notes_kit
    import graphics_kit
    story: list = []
    # Set run dir so graphics_kit helpers know where to store assets
    graphics_kit.set_run_dir(str(run_dir))
    # chdir to run_dir so relative paths (e.g. "assets/page_03.png") resolve
    _orig_cwd = os.getcwd()
    os.chdir(str(run_dir))
    try:
        mod.build(story, notes_kit, notes_kit.STYLES)

        def _content_builder(s: list):
            s.extend(story)

        notes_kit.generate_notes_pdf(
            str(out_path), title, subtitle, _content_builder, theme=theme,
        )
    except Exception as e:
        errors.append(f"BuildError: {type(e).__name__}: {e}")
        _emit({"success": False, "pdf_path": None, "pages": 0,
               "previews": [], "assets": [], "warnings": [], "errors": errors})
        return
    finally:
        os.chdir(_orig_cwd)

    # 6. Collect graphics warnings
    try:
        gfx_warns = notes_kit.get_graphics_warnings()
        warnings.extend(gfx_warns)
    except Exception:
        pass

    # 7. Page count
    page_count = 0
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(out_path))
        page_count = len(reader.pages)
    except Exception as e:
        warnings.append(f"Could not read page count: {e}")

    # 8. Preview PNGs (always generate for all pages)
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(str(out_path), dpi=120)
        for i, page_img in enumerate(pages):
            preview_path = out_path.parent / f"preview_p{i + 1}.png"
            page_img.save(str(preview_path))
            previews.append(str(preview_path.relative_to(REPO_ROOT)))
    except Exception as e:
        warnings.append(f"Preview generation failed: {e}")

    # 9. Collect asset files
    if assets_dir.exists():
        for f in sorted(assets_dir.iterdir()):
            if f.suffix.lower() == ".png":
                asset_files.append(str(f.relative_to(REPO_ROOT)))

    _emit({
        "success": True,
        "pdf_path": str(out_path.relative_to(REPO_ROOT)),
        "pages": page_count,
        "previews": previews,
        "assets": asset_files,
        "warnings": warnings,
        "errors": errors,
    })


# ---------------------------------------------------------------------------
# build (detailed mode — book.json manifest)
# ---------------------------------------------------------------------------

def cmd_build_detailed(args: argparse.Namespace):
    content_path = Path(args.content).resolve()
    out_path = Path(args.out).resolve()
    title = args.title or "Notes"
    subtitle = args.subtitle
    preview = args.preview
    theme = args.theme or "light"
    no_cache = args.no_cache
    only_ids = args.only
    skip_ids = args.skip

    errors: list[str] = []
    warnings: list[str] = []
    previews: list[str] = []
    asset_files: list[str] = []
    chapters_rendered = 0
    chapters_failed: list[str] = []

    # 1. Sanity checks
    if not content_path.exists():
        errors.append(f"Manifest file not found: {content_path}")
        _emit({"success": False, "pdf_path": None, "pages": 0,
               "previews": [], "assets": [], "warnings": [],
               "errors": errors, "chapters_rendered": 0, "chapters_failed": []})
        return

    if content_path.suffix not in (".json",):
        errors.append(f"Manifest file must be .json, got: {content_path.suffix}")
        _emit({"success": False, "pdf_path": None, "pages": 0,
               "previews": [], "assets": [], "warnings": [],
               "errors": errors, "chapters_rendered": 0, "chapters_failed": []})
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Load manifest
    manifest_dir = content_path.parent
    try:
        raw = json.loads(content_path.read_text())
    except json.JSONDecodeError as e:
        _emit({"success": False, "pdf_path": None, "pages": 0,
               "previews": [], "assets": [], "warnings": [],
               "errors": [f"Invalid JSON in manifest: {e}"],
               "chapters_rendered": 0, "chapters_failed": []})
        return

    # Validate required fields
    if "chapters" not in raw or not isinstance(raw["chapters"], list):
        _emit({"success": False, "pdf_path": None, "pages": 0,
               "previews": [], "assets": [], "warnings": [],
               "errors": ["book.json must contain a 'chapters' array"],
               "chapters_rendered": 0, "chapters_failed": []})
        return

    manifest_title = raw.get("title", title)
    manifest_subtitle = raw.get("subtitle", subtitle)
    manifest_theme = raw.get("theme", theme)
    chapters = raw["chapters"]

    # CLI overrides manifest if explicitly provided
    final_title = args.title or manifest_title or "Notes"
    final_subtitle = args.subtitle if args.subtitle is not None else manifest_subtitle
    final_theme = args.theme if args.theme is not None else manifest_theme

    # Filter chapters by --only / --skip
    if only_ids:
        only_set = set(only_ids.split(","))
        chapters = [ch for ch in chapters if ch.get("id") in only_set]
        unknown = only_set - {ch.get("id") for ch in chapters}
        if unknown:
            err = f"Unknown chapter id(s) in --only: {unknown}. Known: {[c.get('id') for c in raw['chapters']]}"
            _emit({"success": False, "pdf_path": None, "pages": 0,
                   "previews": [], "assets": [], "warnings": [],
                   "errors": [err], "chapters_rendered": 0, "chapters_failed": []})
            return

    if skip_ids:
        skip_set = set(skip_ids.split(","))
        chapters = [ch for ch in chapters if ch.get("id") not in skip_set]

    # 3. Setup assets dir
    run_dir = out_path.parent
    assets_dir = run_dir / "assets"
    if no_cache:
        _wipe_kit_cache(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Clean old preview PNGs
    for old_preview in run_dir.glob("preview_p*.png"):
        old_preview.unlink()

    # 4. Set matplotlib backend
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass

    # 5. Chapter rendering loop
    import notes_kit
    import graphics_kit
    story: list = []
    graphics_kit.set_run_dir(str(run_dir))

    # chdir to run_dir so relative paths (e.g. "assets/page_03.png") resolve
    _orig_cwd = os.getcwd()
    os.chdir(str(run_dir))

    for i, ch in enumerate(chapters):
        ch_id = ch.get("id", f"ch{str(i).zfill(2)}")
        ch_file_rel = ch.get("file", "")
        ch_file = (manifest_dir / ch_file_rel).resolve()

        # a. Verify file exists
        if not ch_file.exists():
            warnings.append(f"Chapter '{ch_id}': file not found: {ch_file_rel}")
            notes_kit.add_callout(
                story,
                f"Chapter <b>{ch.get('title', ch_id)}</b> file not found: {ch_file_rel}",
                kind="warning",
            )
            chapters_failed.append(ch_id)
            continue

        # b. Dynamic import
        try:
            spec = importlib.util.spec_from_file_location(
                f"detail_{ch_id}", str(ch_file)
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            warnings.append(f"Chapter '{ch_id}': import failed: {e}")
            notes_kit.add_callout(
                story,
                f"Chapter <b>{ch.get('title', ch_id)}</b> failed to load: {type(e).__name__}: {e}",
                kind="warning",
            )
            chapters_failed.append(ch_id)
            continue

        # c. Confirm build callable
        if not hasattr(mod, "build") or not callable(mod.build):
            warnings.append(f"Chapter '{ch_id}': missing build() function")
            notes_kit.add_callout(
                story,
                f"Chapter <b>{ch.get('title', ch_id)}</b>: missing build(story, kit, STYLES) function.",
                kind="warning",
            )
            chapters_failed.append(ch_id)
            continue

        # d. Page break before chapter (unless first chapter)
        starts_on_new_page = getattr(mod, "starts_on_new_page", True)
        if i > 0 and starts_on_new_page:
            notes_kit.add_page_break(story)

        # e. Auto title block
        suppress_chapter_title = getattr(mod, "suppress_chapter_title", False)
        if not suppress_chapter_title:
            ch_title = getattr(mod, "chapter_title_override", None) or ch.get("title")
            ch_subtitle = ch.get("subtitle")
            if ch_title:
                notes_kit.add_title_block(story, ch_title, ch_subtitle)

        # f. Divider after title
        show_chapter_divider = getattr(mod, "show_chapter_divider", False)
        if show_chapter_divider:
            notes_kit.add_divider(story)

        # g. Call chapter's build()
        MAX_CHAPTER_RETRIES = 3
        last_err = None
        for attempt in range(1, MAX_CHAPTER_RETRIES + 1):
            story_before = len(story)
            try:
                mod.build(story, notes_kit, notes_kit.STYLES)
                chapters_rendered += 1
                break
            except Exception as e:
                # Roll back partial story so retry starts clean
                del story[story_before:]
                last_err = e
                if attempt < MAX_CHAPTER_RETRIES:
                    # Re-import module so agent file edits take effect
                    try:
                        spec = importlib.util.spec_from_file_location(
                            f"detail_{ch_id}_r{attempt}", str(ch_file)
                        )
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                    except Exception:
                        pass  # keep old mod if re-import fails
                    continue
                # Final failure — visible callout + warning
                notes_kit.add_callout(
                    story,
                    f"Chapter <b>{ch.get('title', ch_id)}</b> failed after "
                    f"{MAX_CHAPTER_RETRIES} attempts: {type(last_err).__name__}: {last_err}",
                    kind="warning",
                )
                chapters_failed.append(ch_id)

    # Check for unreferenced .py files in chapters/
    chapters_dir = manifest_dir / "chapters"
    if chapters_dir.exists():
        referenced_files = set()
        for ch in chapters:
            ref = (manifest_dir / ch.get("file", "")).resolve()
            referenced_files.add(ref)
        for py_file in chapters_dir.rglob("*.py"):
            if py_file.resolve() not in referenced_files:
                warnings.append(
                    f"Unreferenced chapter file: {py_file.relative_to(manifest_dir)}"
                )

    # 6. Generate PDF
    try:
        def _content_builder(s: list):
            s.extend(story)

        notes_kit.generate_notes_pdf(
            str(out_path), final_title, final_subtitle, _content_builder,
            theme=final_theme,
        )
    except Exception as e:
        errors.append(f"BuildError: {type(e).__name__}: {e}")
        _emit({
            "success": False, "pdf_path": None, "pages": 0,
            "previews": [], "assets": [], "warnings": warnings,
            "errors": errors,
            "chapters_rendered": chapters_rendered,
            "chapters_failed": chapters_failed,
        })
        return
    finally:
        os.chdir(_orig_cwd)

    # 7. Collect graphics warnings
    try:
        gfx_warns = notes_kit.get_graphics_warnings()
        warnings.extend(gfx_warns)
    except Exception:
        pass

    # 8. Page count
    page_count = 0
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(out_path))
        page_count = len(reader.pages)
    except Exception as e:
        warnings.append(f"Could not read page count: {e}")

    # 9. Preview PNGs + chapter page mapping
    # Always generate previews for all pages
    chapter_pages: dict[str, dict] = {}
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(str(out_path), dpi=120)
        for i, page_img in enumerate(pages):
            preview_path = out_path.parent / f"preview_p{i + 1}.png"
            page_img.save(str(preview_path))
            previews.append(str(preview_path.relative_to(REPO_ROOT)))
    except Exception as e:
        warnings.append(f"Preview generation failed: {e}")

    # Map chapters to page ranges by searching for chapter titles in PDF text
    try:
        import pdfplumber
        with pdfplumber.open(str(out_path)) as pdf:
            page_texts = [(i + 1, (p.extract_text() or "")) for i, p in enumerate(pdf.pages)]

            # Build chapter title → page mapping
            ch_titles = []
            for ch in raw.get("chapters", []):
                ch_id = ch.get("id", "")
                ch_title = ch.get("title", "")
                if ch_id and ch_title:
                    ch_titles.append((ch_id, ch_title))

            for idx, (ch_id, ch_title) in enumerate(ch_titles):
                start_page = None
                # Search for chapter title on each page
                for page_num, text in page_texts:
                    if ch_title in text:
                        start_page = page_num
                        break

                if start_page is not None:
                    # End page = next chapter's start page - 1, or last page
                    if idx + 1 < len(ch_titles):
                        next_title = ch_titles[idx + 1][1]
                        end_page = len(page_texts)
                        for page_num, text in page_texts:
                            if page_num > start_page and next_title in text:
                                end_page = page_num - 1
                                break
                    else:
                        end_page = len(page_texts)

                    chapter_pages[ch_id] = {
                        "start": start_page,
                        "end": end_page,
                        "pages": list(range(start_page, end_page + 1)),
                    }
    except Exception as e:
        warnings.append(f"Chapter page mapping failed: {e}")

    # 10. Collect asset files
    if assets_dir.exists():
        for f in sorted(assets_dir.iterdir()):
            if f.suffix.lower() == ".png":
                asset_files.append(str(f.relative_to(REPO_ROOT)))

    _emit({
        "success": True,
        "pdf_path": str(out_path.relative_to(REPO_ROOT)),
        "pages": page_count,
        "previews": previews,
        "assets": asset_files,
        "warnings": warnings,
        "errors": errors,
        "chapters_rendered": chapters_rendered,
        "chapters_failed": chapters_failed,
        "chapter_pages": chapter_pages,
    })


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace):
    pdf_path = Path(args.pdf).resolve()
    errors: list[str] = []
    issues: list[dict] = []

    if not pdf_path.exists():
        _emit({"success": False, "issues": [],
               "errors": [f"PDF not found: {pdf_path}"]})
        return

    try:
        import pdfplumber
    except ImportError:
        _emit({"success": False, "issues": [],
               "errors": ["pdfplumber is required for validation. Install with: uv add pdfplumber"]})
        return

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
    except Exception as e:
        _emit({"success": False, "issues": [],
               "errors": [f"Cannot read PDF: {e}"]})
        return

    from notes_kit import PAGE_SIZE, MARGIN
    frame_right = PAGE_SIZE[0] - MARGIN + 1  # 1pt tolerance
    frame_left = MARGIN - 1

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1

                # --- orphan_heading ---
                words = page.extract_words() if hasattr(page, "extract_words") else []
                if words:
                    bottom_words = [w for w in words if w.get("bottom", 0) > page.height * 0.85]
                    if bottom_words:
                        bottom_text = " ".join(w.get("text", "") for w in bottom_words)
                        if len(bottom_text) < 80 and not bottom_text.endswith((".", ",", ";", ":")):
                            issues.append({
                                "kind": "orphan_heading",
                                "page": page_num,
                                "severity": "warn",
                                "detail": (
                                    f"Short text at page bottom ({len(bottom_text)} chars). "
                                    "Consider wrapping with add_keep_together."
                                ),
                            })

                # --- text_touches_margin ---
                for word in words:
                    x0 = word.get("x0", 0)
                    x1 = word.get("x1", 0)
                    if x1 > frame_right:
                        issues.append({
                            "kind": "text_touches_margin",
                            "page": page_num,
                            "severity": "error",
                            "detail": f"Word '{word.get('text', '')}' extends past right margin (x1={x1:.0f}).",
                        })
                    if x0 < frame_left:
                        issues.append({
                            "kind": "text_touches_margin",
                            "page": page_num,
                            "severity": "error",
                            "detail": f"Word '{word.get('text', '')}' extends past left margin (x0={x0:.0f}).",
                        })

                # --- code_overflow ---
                for table in (page.find_tables() if hasattr(page, "find_tables") else []):
                    bbox = table.bbox
                    if bbox[2] > frame_right:
                        issues.append({
                            "kind": "code_overflow",
                            "page": page_num,
                            "severity": "error",
                            "detail": f"Table/code block extends past right margin (x1={bbox[2]:.0f}).",
                        })

                # --- image_oversized ---
                for img in (page.images if hasattr(page, "images") else []):
                    img_right = img.get("x1", 0)
                    if img_right > frame_right:
                        issues.append({
                            "kind": "image_oversized",
                            "page": page_num,
                            "severity": "error",
                            "detail": f"Image extends past right margin (x1={img_right:.0f}, frame={frame_right:.0f}).",
                        })

                # --- empty_page ---
                text = page.extract_text() or ""
                images = page.images if hasattr(page, "images") else []
                if len(text.strip()) < 30 and len(images) < 2:
                    issues.append({
                        "kind": "empty_page",
                        "page": page_num,
                        "severity": "warn",
                        "detail": "Page has very little content (possible empty trailing page).",
                    })

    except Exception as e:
        errors.append(f"Validation error: {type(e).__name__}: {e}")

    _emit({
        "success": True,
        "issues": issues,
        "errors": errors,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="runner",
        description="PDF notes skill runner — build, validate, and scaffold notes-as-code.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Scaffold a workspace/run<N>/content.py or book.json")
    p_init.add_argument("--name", type=str, default=None, help="Optional run name hint")
    p_init.add_argument("--mode", type=str, default="single",
                        choices=["single", "detailed"],
                        help="Scaffold mode: single (content.py) or detailed (book.json + chapters/)")

    # build
    p_build = sub.add_parser("build", help="Build notes.pdf from content.py or book.json")
    p_build.add_argument("--content", required=True, help="Path to content.py or book.json")
    p_build.add_argument("--out", required=True, help="Output PDF path")
    p_build.add_argument("--title", type=str, default="Notes", help="Document title")
    p_build.add_argument("--subtitle", type=str, default=None, help="Document subtitle")
    p_build.add_argument("--preview", action="store_true",
                         help="(Deprecated) Previews are now always generated for all pages")
    p_build.add_argument("--theme", type=str, default="light", help="Theme: light | dark")
    p_build.add_argument("--no-cache", action="store_true",
                         help="Wipe assets/ before building (force re-render all graphics)")
    p_build.add_argument("--only", type=str, default=None,
                         help="Comma-separated chapter ids to render (detailed mode only)")
    p_build.add_argument("--skip", type=str, default=None,
                         help="Comma-separated chapter ids to exclude (detailed mode only)")

    # validate
    p_val = sub.add_parser("validate", help="Validate a rendered PDF for layout issues")
    p_val.add_argument("--pdf", required=True, help="Path to the PDF to validate")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "build":
        content_path = Path(args.content).resolve()
        if content_path.suffix == ".json":
            cmd_build_detailed(args)
        else:
            cmd_build(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
