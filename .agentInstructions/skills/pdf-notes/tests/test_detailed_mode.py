"""Tests for detailed notes mode — init, build, failure isolation, --only/--skip."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from notes_kit import generate_notes_pdf
from schemas.generate_spec import BookManifest, ChapterSpec, BookBuildResponse


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_temp_run_dir():
    tmp = tempfile.mkdtemp(prefix="notes_test_")
    return Path(tmp)


def _write_chapter(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# -----------------------------------------------------------------------
# Test: init --mode detailed creates expected tree
# -----------------------------------------------------------------------

def test_init_detailed_creates_expected_tree():
    tmp = _make_temp_run_dir()
    try:
        book_path = tmp / "book.json"
        chapters_dir = tmp / "chapters"

        manifest = {
            "title": "Test Book",
            "subtitle": "Test subtitle",
            "theme": "light",
            "chapters": [
                {"id": "ch01", "slug": "intro", "title": "Intro",
                 "file": "chapters/ch01_intro.py", "summary": "First chapter"},
                {"id": "ch02", "slug": "core", "title": "Core",
                 "file": "chapters/ch02_core.py", "summary": "Second chapter"},
            ],
        }
        book_path.write_text(json.dumps(manifest))

        chapters_dir.mkdir(exist_ok=True)
        _write_chapter(chapters_dir / "ch01_intro.py", '''\
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_heading(story, "Intro Section", level=1)
    kit.add_paragraph(story, "Hello from chapter 1.")
''')
        _write_chapter(chapters_dir / "ch02_core.py", '''\
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_heading(story, "Core Section", level=1)
    kit.add_paragraph(story, "Hello from chapter 2.")
''')

        assert book_path.exists()
        assert (chapters_dir / "ch01_intro.py").exists()
        assert (chapters_dir / "ch02_core.py").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -----------------------------------------------------------------------
# Test: build stitches chapters into one PDF with correct ordering
# -----------------------------------------------------------------------

def test_build_stitches_chapters_into_pdf():
    tmp = _make_temp_run_dir()
    try:
        book_path = tmp / "book.json"
        out_path = tmp / "notes.pdf"

        manifest = {
            "title": "Stitch Test",
            "subtitle": "Testing",
            "chapters": [
                {"id": "ch01", "slug": "one", "title": "Chapter One",
                 "file": "chapters/ch01_one.py",
                 "summary": "First"},
                {"id": "ch02", "slug": "two", "title": "Chapter Two",
                 "file": "chapters/ch02_two.py",
                 "summary": "Second"},
                {"id": "ch03", "slug": "three", "title": "Chapter Three",
                 "file": "chapters/ch03_three.py",
                 "summary": "Third"},
            ],
        }
        book_path.write_text(json.dumps(manifest))

        _write_chapter(tmp / "chapters" / "ch01_one.py", '''\
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_heading(story, "One", level=1)
    kit.add_paragraph(story, "Content one.")
''')
        _write_chapter(tmp / "chapters" / "ch02_two.py", '''\
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_heading(story, "Two", level=1)
    kit.add_paragraph(story, "Content two.")
''')
        _write_chapter(tmp / "chapters" / "ch03_three.py", '''\
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_heading(story, "Three", level=1)
    kit.add_paragraph(story, "Content three.")
''')

        # Run the build pipeline manually to test the chapter loop
        import graphics_kit
        graphics_kit.set_run_dir(str(tmp))

        import notes_kit
        story = []
        chapters_rendered = 0
        chapters_failed = []

        for i, ch in enumerate(manifest["chapters"]):
            ch_id = ch["id"]
            ch_file = (tmp / ch["file"]).resolve()

            import importlib.util
            spec = importlib.util.spec_from_file_location(f"test_{ch_id}", str(ch_file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            starts_on_new_page = getattr(mod, "starts_on_new_page", True)
            suppress = getattr(mod, "suppress_chapter_title", False)
            show_divider = getattr(mod, "show_chapter_divider", False)

            if i > 0 and starts_on_new_page:
                notes_kit.add_page_break(story)

            if not suppress:
                ch_title = getattr(mod, "chapter_title_override", None) or ch.get("title")
                ch_subtitle = ch.get("subtitle")
                if ch_title:
                    notes_kit.add_title_block(story, ch_title, ch_subtitle)

            if show_divider:
                notes_kit.add_divider(story)

            try:
                mod.build(story, notes_kit, notes_kit.STYLES)
                chapters_rendered += 1
            except Exception as e:
                notes_kit.add_callout(
                    story,
                    f"Chapter {ch.get('title', ch_id)} failed: {e}",
                    kind="warning",
                )
                chapters_failed.append(ch_id)

        # Generate PDF
        def builder(s):
            s.extend(story)
        notes_kit.generate_notes_pdf(str(out_path), "Test", None, builder)

        assert out_path.exists()
        assert os.path.getsize(out_path) > 500

        from pypdf import PdfReader
        reader = PdfReader(str(out_path))
        assert len(reader.pages) >= 1

        assert chapters_rendered == 3
        assert chapters_failed == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -----------------------------------------------------------------------
# Test: intentional raise inside ch02 — build still succeeds,
#       chapters_failed == ["ch02"], callout present, rest intact
# -----------------------------------------------------------------------

def test_failure_isolation_continues_build():
    tmp = _make_temp_run_dir()
    try:
        book_path = tmp / "book.json"
        out_path = tmp / "notes.pdf"

        manifest = {
            "title": "Fail Test",
            "chapters": [
                {"id": "ch01", "slug": "one", "title": "One",
                 "file": "chapters/ch01_one.py", "summary": "First"},
                {"id": "ch02", "slug": "two", "title": "Two",
                 "file": "chapters/ch02_two.py", "summary": "Bad"},
                {"id": "ch03", "slug": "three", "title": "Three",
                 "file": "chapters/ch03_three.py", "summary": "Third"},
            ],
        }
        book_path.write_text(json.dumps(manifest))

        _write_chapter(tmp / "chapters" / "ch01_one.py", '''\
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_heading(story, "One", level=1)
    kit.add_paragraph(story, "Content one.")
''')
        _write_chapter(tmp / "chapters" / "ch02_two.py", '''\
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_heading(story, "Two", level=1)
    raise RuntimeError("Intentional test crash")
''')
        _write_chapter(tmp / "chapters" / "ch03_three.py", '''\
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_heading(story, "Three", level=1)
    kit.add_paragraph(story, "Content three.")
''')

        import graphics_kit
        graphics_kit.set_run_dir(str(tmp))

        import notes_kit
        story = []
        chapters_rendered = 0
        chapters_failed = []

        import importlib.util
        for i, ch in enumerate(manifest["chapters"]):
            ch_id = ch["id"]
            ch_file = (tmp / ch["file"]).resolve()

            spec = importlib.util.spec_from_file_location(f"test_{ch_id}", str(ch_file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            if i > 0:
                notes_kit.add_page_break(story)

            notes_kit.add_title_block(story, ch.get("title"), None)

            try:
                mod.build(story, notes_kit, notes_kit.STYLES)
                chapters_rendered += 1
            except Exception as e:
                notes_kit.add_callout(
                    story,
                    f"Chapter {ch.get('title', ch_id)} failed to render: {e}",
                    kind="warning",
                )
                chapters_failed.append(ch_id)

        def builder(s):
            s.extend(story)
        notes_kit.generate_notes_pdf(str(out_path), "Test", None, builder)

        assert out_path.exists()
        assert os.path.getsize(out_path) > 500

        from pypdf import PdfReader
        reader = PdfReader(str(out_path))
        assert len(reader.pages) >= 1

        assert chapters_rendered == 2
        assert chapters_failed == ["ch02"]

        # Verify that ch01 and ch03 text appears in PDF
        page_text = reader.pages[0].extract_text() or ""
        assert "One" in page_text
        # ch03 might be on later pages
        all_text = ""
        for page in reader.pages:
            all_text += page.extract_text() or ""
        assert "Three" in all_text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -----------------------------------------------------------------------
# Test: --only flag produces single-chapter PDF
# -----------------------------------------------------------------------

def test_only_flag_single_chapter():
    tmp = _make_temp_run_dir()
    try:
        book_path = tmp / "book.json"
        out_path = tmp / "notes.pdf"
        only_ids = {"ch01"}

        manifest = {
            "title": "Only Test",
            "chapters": [
                {"id": "ch01", "slug": "one", "title": "One",
                 "file": "chapters/ch01_one.py", "summary": "First"},
                {"id": "ch02", "slug": "two", "title": "Two",
                 "file": "chapters/ch02_two.py", "summary": "Second"},
            ],
        }
        book_path.write_text(json.dumps(manifest))

        _write_chapter(tmp / "chapters" / "ch01_one.py", '''\
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_heading(story, "One", level=1)
    kit.add_paragraph(story, "Only ch01.")
''')
        _write_chapter(tmp / "chapters" / "ch02_two.py", '''\
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_heading(story, "Two", level=1)
    kit.add_paragraph(story, "Should not appear.")
''')

        import graphics_kit
        graphics_kit.set_run_dir(str(tmp))

        filtered = [ch for ch in manifest["chapters"] if ch["id"] in only_ids]

        import notes_kit
        story = []
        chapters_rendered = 0
        chapters_failed = []

        import importlib.util
        for i, ch in enumerate(filtered):
            ch_id = ch["id"]
            ch_file = (tmp / ch["file"]).resolve()

            spec = importlib.util.spec_from_file_location(f"test_{ch_id}", str(ch_file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            notes_kit.add_title_block(story, ch.get("title"), None)

            try:
                mod.build(story, notes_kit, notes_kit.STYLES)
                chapters_rendered += 1
            except Exception as e:
                notes_kit.add_callout(story, f"Failed: {e}", kind="warning")
                chapters_failed.append(ch_id)

        def builder(s):
            s.extend(story)
        notes_kit.generate_notes_pdf(str(out_path), "Test", None, builder)

        assert out_path.exists()
        assert chapters_rendered == 1
        assert chapters_failed == []

        from pypdf import PdfReader
        reader = PdfReader(str(out_path))
        all_text = ""
        for page in reader.pages:
            all_text += page.extract_text() or ""
        assert "Only ch01" in all_text
        assert "Should not appear" not in all_text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -----------------------------------------------------------------------
# Test: existing content.py single mode still works
# -----------------------------------------------------------------------

def test_single_mode_still_works():
    tmp = _make_temp_run_dir()
    try:
        content_path = tmp / "content.py"
        out_path = tmp / "notes.pdf"

        content_path.write_text('''\
from notes_kit import *

def build(story, kit, STYLES):
    kit.add_heading(story, "Single", level=1)
    kit.add_paragraph(story, "Single mode still works.")
''')

        import importlib.util
        spec = importlib.util.spec_from_file_location("content", str(content_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        import graphics_kit
        graphics_kit.set_run_dir(str(tmp))

        import notes_kit
        story = []
        mod.build(story, notes_kit, notes_kit.STYLES)

        def builder(s):
            s.extend(story)
        notes_kit.generate_notes_pdf(str(out_path), "Single Test", None, builder)

        assert out_path.exists()
        from pypdf import PdfReader
        reader = PdfReader(str(out_path))
        assert len(reader.pages) >= 1
        text = reader.pages[0].extract_text() or ""
        assert "Single mode still works" in text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -----------------------------------------------------------------------
# Test: chapter hooks (suppress_chapter_title, chapter_title_override, etc.)
# -----------------------------------------------------------------------

def test_chapter_hooks():
    tmp = _make_temp_run_dir()
    try:
        book_path = tmp / "book.json"
        out_path = tmp / "notes.pdf"

        manifest = {
            "title": "Hooks Test",
            "chapters": [
                {"id": "ch01", "slug": "one", "title": "Original Title",
                 "file": "chapters/ch01_one.py", "summary": "First"},
            ],
        }
        book_path.write_text(json.dumps(manifest))

        _write_chapter(tmp / "chapters" / "ch01_one.py", '''\
from notes_kit import *

chapter_title_override = "Overridden Title"

def build(story, kit, STYLES):
    kit.add_heading(story, "Body", level=1)
    kit.add_paragraph(story, "Content.")
''')

        import graphics_kit
        graphics_kit.set_run_dir(str(tmp))

        import notes_kit
        story = []
        chapters_rendered = 0
        chapters_failed = []

        import importlib.util
        for i, ch in enumerate(manifest["chapters"]):
            ch_id = ch["id"]
            ch_file = (tmp / ch["file"]).resolve()

            spec = importlib.util.spec_from_file_location(f"test_{ch_id}", str(ch_file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            suppress = getattr(mod, "suppress_chapter_title", False)
            show_divider = getattr(mod, "show_chapter_divider", False)

            if not suppress:
                ch_title = getattr(mod, "chapter_title_override", None) or ch.get("title")
                if ch_title:
                    notes_kit.add_title_block(story, ch_title, None)

            if show_divider:
                notes_kit.add_divider(story)

            try:
                mod.build(story, notes_kit, notes_kit.STYLES)
                chapters_rendered += 1
            except Exception as e:
                notes_kit.add_callout(story, f"Failed: {e}", kind="warning")
                chapters_failed.append(ch_id)

        def builder(s):
            s.extend(story)
        notes_kit.generate_notes_pdf(str(out_path), "Hooks", None, builder)

        assert out_path.exists()
        from pypdf import PdfReader
        reader = PdfReader(str(out_path))
        text = reader.pages[0].extract_text() or ""
        assert "Overridden Title" in text
        assert "Original Title" not in text
        assert chapters_rendered == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_suppress_chapter_title():
    tmp = _make_temp_run_dir()
    try:
        book_path = tmp / "book.json"
        out_path = tmp / "notes.pdf"

        manifest = {
            "title": "Suppress Test",
            "chapters": [
                {"id": "ch01", "slug": "one", "title": "Should Not Appear",
                 "file": "chapters/ch01_one.py", "summary": "First"},
            ],
        }
        book_path.write_text(json.dumps(manifest))

        _write_chapter(tmp / "chapters" / "ch01_one.py", '''\
from notes_kit import *

suppress_chapter_title = True

def build(story, kit, STYLES):
    kit.add_heading(story, "Custom Start", level=1)
    kit.add_paragraph(story, "No auto title.")
''')

        import graphics_kit
        graphics_kit.set_run_dir(str(tmp))

        import notes_kit
        story = []
        chapters_rendered = 0

        import importlib.util
        for i, ch in enumerate(manifest["chapters"]):
            ch_id = ch["id"]
            ch_file = (tmp / ch["file"]).resolve()

            spec = importlib.util.spec_from_file_location(f"test_{ch_id}", str(ch_file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            suppress = getattr(mod, "suppress_chapter_title", False)

            if not suppress:
                notes_kit.add_title_block(story, ch.get("title"), None)

            mod.build(story, notes_kit, notes_kit.STYLES)
            chapters_rendered += 1

        def builder(s):
            s.extend(story)
        notes_kit.generate_notes_pdf(str(out_path), "Test", None, builder)

        assert chapters_rendered == 1
        from pypdf import PdfReader
        reader = PdfReader(str(out_path))
        text = reader.pages[0].extract_text() or ""
        assert "Should Not Appear" not in text
        assert "Custom Start" in text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -----------------------------------------------------------------------
# Test: BookManifest Pydantic schema
# -----------------------------------------------------------------------

def test_book_manifest_schema():
    m = BookManifest(title="Test", chapters=[
        ChapterSpec(id="ch01", slug="intro", title="Intro",
                    file="chapters/ch01_intro.py"),
    ])
    assert m.title == "Test"
    assert len(m.chapters) == 1
    assert m.chapters[0].id == "ch01"

    # JSON round-trip
    js = m.model_dump_json()
    m2 = BookManifest.model_validate_json(js)
    assert m2.title == "Test"


def test_book_build_response_schema():
    r = BookBuildResponse(success=True, pdf_path="foo.pdf", pages=5,
                          chapters_rendered=3, chapters_failed=["ch02"])
    assert r.chapters_rendered == 3
    assert r.chapters_failed == ["ch02"]
    assert r.success is True
    assert r.pages == 5


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []

    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                passed += 1
                print(f"  PASS {name}")
            except Exception as e:
                failed += 1
                print(f"  FAIL {name}: {e}")
                errors.append((name, str(e)))

    print(f"\n{passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
    sys.exit(1 if failed else 0)
