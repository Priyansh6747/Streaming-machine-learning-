"""Tests for highlighting.py — whitespace preservation, dedent, line numbers, fallback."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from reportlab.lib.styles import ParagraphStyle

from highlighting import (
    highlight_to_reportlab,
    highlight_to_table,
    _fallback_xml,
    _autoformat,
)

_CODE_STYLE = ParagraphStyle("TestCode", fontName="Courier", fontSize=9)
_LINE_NUM_STYLE = ParagraphStyle("TestLineNum", fontName="Courier", fontSize=9)


NESTED_PY = "def f(n):\n    x = 1\n    if x:\n        return 2\n"


def test_indentation_preserved():
    xml = highlight_to_reportlab(NESTED_PY, "python")
    lines = xml.split("<br/>")
    deepest = lines[3]  # "        return 2"
    assert deepest.startswith("<font")
    # 8 leading spaces -> 8 &nbsp;
    stripped = deepest.split(">", 1)[1]
    assert stripped.startswith("&nbsp;" * 8)
    for line in lines:
        assert not line.startswith(" ")


def test_tabs_expanded():
    tabbed = "def f():\n\tx = 1\n\treturn x\n"
    xml = highlight_to_reportlab(tabbed, "python")
    assert "\t" not in xml


def test_uniform_indent_dedented():
    indented = "    def f():\n        return 1\n"
    flush = "def f():\n    return 1\n"
    assert highlight_to_reportlab(indented, "python") == highlight_to_reportlab(flush, "python")


def test_line_numbers_match():
    code = "\n".join(f"line{i}" for i in range(14))
    table = highlight_to_table(code, "text", code_style=_CODE_STYLE, line_num_style=_LINE_NUM_STYLE)
    assert len(table._cellvalues) == 14


def test_no_pygments_fallback():
    xml = _fallback_xml("    if x:\n        y = 1\n")
    lines = xml.split("<br/>")
    assert lines[0].startswith("&nbsp;" * 4)
    assert lines[1].startswith("&nbsp;" * 8)


def test_autoformat_noop_on_bad_syntax():
    bad = "def f("
    assert _autoformat(bad, "python") == bad


def test_autoformat_formats_valid_python():
    pytest.importorskip("black")
    messy = "def f(x,y):\n    return x+y\n"
    formatted = _autoformat(messy, "python")
    assert "x + y" in formatted
