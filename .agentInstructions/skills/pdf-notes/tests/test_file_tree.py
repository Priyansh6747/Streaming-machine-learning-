"""Tests for file_tree module — validation, schema errors, rendering, and splitting."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate

from file_tree import (
    IconTreeBlock,
    _count_rows,
    _flatten,
    validate_file_tree_spec,
    split_tree_spec,
)
from notes_kit import (
    add_file_tree,
    generate_notes_pdf,
    PAGE_SIZE,
    MARGIN,
)

# -----------------------------------------------------------------------
# Test fixtures — matching the required nested examples
# -----------------------------------------------------------------------

DRIVE_C = {
    "label": "C:\\", "node_type": "drive", "children": [
        {"label": "Program Files", "node_type": "folder", "children": [
            {"label": "...", "node_type": "ellipsis"},
            {"label": "...", "node_type": "ellipsis"},
        ]},
        {"label": "Windows", "node_type": "folder", "children": [
            {"label": "...", "node_type": "ellipsis"},
            {"label": "...", "node_type": "ellipsis"},
        ]},
        {"label": "...", "node_type": "ellipsis"},
        {"label": "...", "node_type": "ellipsis"},
    ],
}

DRIVE_D = {
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
        {"label": "...", "node_type": "ellipsis"},
    ],
}

# -----------------------------------------------------------------------
# Validation tests
# -----------------------------------------------------------------------

def test_validate_valid_specs():
    assert validate_file_tree_spec(DRIVE_C) is None
    assert validate_file_tree_spec(DRIVE_D) is None
    # Minimal valid single node
    assert validate_file_tree_spec({"label": "root", "node_type": "folder"}) is None
    # Ellipsis without label (should default to valid)
    assert validate_file_tree_spec({"node_type": "ellipsis"}) is None


def test_validate_invalid_node_type():
    err = validate_file_tree_spec({"label": "x", "node_type": "garbage"})
    assert err is not None
    assert "garbage" in err
    assert "Invalid node_type" in err

    err = validate_file_tree_spec({"label": "nested", "node_type": "drive", "children": [
        {"label": "sub", "node_type": "invalid_type"}
    ]})
    assert "invalid_type" in err
    assert ".children[0]" in err


def test_validate_missing_label():
    err = validate_file_tree_spec({"node_type": "file"})
    assert err is not None
    assert "Missing required 'label'" in err
    assert "file" in err

    err = validate_file_tree_spec({"label": "root", "node_type": "folder", "children": [
        {"node_type": "file"}
    ]})
    assert "Missing required 'label'" in err
    assert ".children[0]" in err


def test_validate_not_a_dict():
    err = validate_file_tree_spec([])  # type: ignore
    assert err is not None
    assert "must be a dict" in err


def test_validate_bad_children():
    err = validate_file_tree_spec({"label": "x", "node_type": "folder", "children": "not_a_list"})
    assert err is not None
    assert "must be a list" in err


# -----------------------------------------------------------------------
# Row counting and flattening tests
# -----------------------------------------------------------------------

def test_count_rows():
    assert _count_rows(DRIVE_C) == 9
    assert _count_rows(DRIVE_D) == 10
    assert _count_rows({"label": "x", "node_type": "file"}) == 1
    assert _count_rows({"node_type": "ellipsis", "children": []}) == 1


def test_flatten_preserves_order():
    rows = []
    _flatten(DRIVE_D, rows, 0, [])
    labels = [r[1].get("label", "...") for r in rows]
    assert labels[:4] == ["D:\\", "Java", "beginner", "Hello.java"]
    assert rows[0][0] == 0  # depth 0
    assert rows[1][0] == 1  # depth 1
    assert rows[2][0] == 2  # depth 2
    assert rows[3][0] == 3  # depth 3


def test_flatten_ancestor_has_more():
    rows = []
    _flatten(DRIVE_C, rows, 0, [])
    # Program Files (depth 1): ancestor [True] because Windows follows
    prog_files_row = rows[1]
    assert prog_files_row[2] == [True]

    # Windows (depth 1): ancestor [True] because 2 top-level ellipsis follow
    windows_row = rows[4]
    assert windows_row[2] == [True]

    # Last ellipsis leaf at top level (depth 1): ancestor [False]
    last_row = rows[8]
    assert last_row[2] == [False]


# -----------------------------------------------------------------------
# Tree splitting tests
# -----------------------------------------------------------------------

def test_split_no_split_needed():
    blocks = split_tree_spec(DRIVE_D, max_rows_per_block=50)
    assert len(blocks) == 1

    blocks = split_tree_spec(DRIVE_C, max_rows_per_block=50)
    assert len(blocks) == 1


def test_split_small_max_rows():
    blocks = split_tree_spec(DRIVE_D, max_rows_per_block=4)
    assert len(blocks) >= 2

    for block in blocks:
        err = validate_file_tree_spec(block)
        assert err is None, f"Split produced invalid block: {err}"
        assert _count_rows(block) <= 4, f"Block has {_count_rows(block)} rows, exceeds max 4"

    # Verify no corruption — every visible node in original exists in some block
    def all_labels(spec):
        result = [spec.get("label", "...")]
        for child in spec.get("children", []):
            result.extend(all_labels(child))
        return result

    original_labels = all_labels(DRIVE_D)
    block_labels = set()
    for b in blocks:
        for label in all_labels(b):
            block_labels.add(label)

    for label in original_labels:
        assert label in block_labels, f"Label '{label}' missing from split blocks"

    # Verify the top-level structure is preserved: D:\ children are [Java, ..., ...]
    # Java is too large (6 rows), so it recurses. The top-level ellipsis leaves
    # should be in blocks rooted at D:\
    drive_roots = [b for b in blocks if b.get("label") == "D:\\"]
    assert len(drive_roots) > 0, "D:\\ should appear in at least one block"


def test_split_zero_max_rows():
    blocks = split_tree_spec(DRIVE_D, max_rows_per_block=0)
    assert len(blocks) >= 1


# -----------------------------------------------------------------------
# Flowable construction tests
# -----------------------------------------------------------------------

def test_icon_tree_block_wrap_draw():
    from file_tree import VERT_PAD as VP, ROW_HEIGHT as RH
    block = IconTreeBlock(DRIVE_D, width=420)
    w, h = block.wrap(500, 800)
    assert w == 500
    assert h > 0

    expected_h = 2 * VP + RH * 10  # DRIVE_D has 10 rows
    assert h == expected_h


def test_icon_tree_block_zero_children():
    spec = {"label": "root", "node_type": "folder"}
    block = IconTreeBlock(spec)
    w, h = block.wrap(400, 600)
    assert h > 0
    assert w == 400


def test_icon_tree_block_ellipsis():
    spec = {"node_type": "ellipsis"}
    block = IconTreeBlock(spec)
    w, h = block.wrap(300, 500)
    assert h == 2 * 16 + 24 * 1


# -----------------------------------------------------------------------
# Integration: rendering to PDF
# -----------------------------------------------------------------------

def test_pdf_rendering_with_file_trees():
    tmpdir = tempfile.mkdtemp(prefix="notes_filetree_test_")
    try:
        out_path = os.path.join(tmpdir, "test_filetree.pdf")

        def build_content(story):
            spec_c = DRIVE_C
            spec_d = DRIVE_D
            add_file_tree(story, spec_c, caption="Figure 1 — C: drive tree")
            add_file_tree(story, spec_d, caption="Figure 2 — D: drive tree")

        generate_notes_pdf(out_path, "File Tree Test", "Testing icon tree rendering",
                           build_content_fn=build_content)

        assert os.path.isfile(out_path)
        size = os.path.getsize(out_path)
        assert size > 1000, f"PDF too small: {size} bytes"

        from pypdf import PdfReader
        reader = PdfReader(out_path)
        assert len(reader.pages) >= 1
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_pdf_rendering_deep_nesting():
    tmpdir = tempfile.mkdtemp(prefix="notes_filedeep_")
    try:
        out_path = os.path.join(tmpdir, "test_deep_tree.pdf")

        deep_spec = {
            "label": "root", "node_type": "folder", "children": [
                {"label": "level1", "node_type": "folder", "children": [
                    {"label": "level2", "node_type": "folder", "children": [
                        {"label": "level3", "node_type": "folder", "children": [
                            {"label": "file.txt", "node_type": "file"},
                        ]},
                        {"label": "level3b", "node_type": "folder", "children": [
                            {"label": "other.py", "node_type": "file"},
                        ]},
                    ]},
                    {"label": "level2b", "node_type": "file"},
                ]},
            ],
        }

        def build_content(story):
            add_file_tree(story, deep_spec, caption="Figure — 4-level deep nesting")

        generate_notes_pdf(out_path, "Deep Tree Test", "Testing 4-level nesting",
                           build_content_fn=build_content)

        assert os.path.isfile(out_path)
        from pypdf import PdfReader
        reader = PdfReader(out_path)
        assert len(reader.pages) >= 1
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_add_file_tree_invalid_spec_shows_error():
    tmpdir = tempfile.mkdtemp(prefix="notes_treeerr_")
    try:
        out_path = os.path.join(tmpdir, "test_err.pdf")

        def build_content(story):
            add_file_tree(story, {"node_type": "file"})  # missing label

        generate_notes_pdf(out_path, "Error Test", None,
                           build_content_fn=build_content)

        assert os.path.isfile(out_path)
        from pypdf import PdfReader
        reader = PdfReader(out_path)
        # should have a page with an error message, not crash
        assert len(reader.pages) >= 1
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


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
