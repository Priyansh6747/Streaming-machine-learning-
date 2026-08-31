"""highlighting.py — Pygments-to-ReportLab syntax highlighting.

Provides `highlight_to_reportlab()` which turns source code into ReportLab Paragraph
XML with monokai-colored tokens. Used by `add_code(highlight=True)` in notes_kit.

Falls back to plain escaped code if pygments is unavailable or language is unknown.
"""

from __future__ import annotations

import html as _html
import textwrap as _textwrap

# ---------------------------------------------------------------------------
# Monokai color map (token type → hex color)
# ---------------------------------------------------------------------------

_MONOKAI: dict[str, str] = {
    "Token.Keyword":            "#F92672",
    "Token.Keyword.Constant":   "#F92672",
    "Token.Keyword.Declaration":"#F92672",
    "Token.Keyword.Namespace":  "#F92672",
    "Token.Keyword.Pseudo":     "#F92672",
    "Token.Keyword.Reserved":   "#F92672",
    "Token.Keyword.Type":       "#66D9EF",
    "Token.Name.Builtin":       "#66D9EF",
    "Token.Name.Builtin.Pseudo":"#66D9EF",
    "Token.Name.Class":         "#A6E22E",
    "Token.Name.Decorator":     "#A6E22E",
    "Token.Name.Entity":        "#A6E22E",
    "Token.Name.Exception":     "#A6E22E",
    "Token.Name.Function":      "#A6E22E",
    "Token.Name.Label":         "#A6E22E",
    "Token.Name.Tag":           "#F92672",
    "Token.Name.Variable":      "#F8F8F2",
    "Token.Literal.String":     "#E6DB74",
    "Token.Literal.String.Doc": "#E6DB74",
    "Token.Literal.String.Escape":    "#AE81FF",
    "Token.Literal.String.Interpol":"#AE81FF",
    "Token.Literal.Number":     "#AE81FF",
    "Token.Literal.Number.Float":"#AE81FF",
    "Token.Literal.Number.Hex": "#AE81FF",
    "Token.Literal.Number.Integer":"#AE81FF",
    "Token.Literal.Number.Oct": "#AE81FF",
    "Token.Operator":           "#F92672",
    "Token.Operator.Word":      "#F92672",
    "Token.Punctuation":        "#F8F8F2",
    "Token.Comment":            "#75715E",
    "Token.Comment.Single":     "#75715E",
    "Token.Comment.Multiline":  "#75715E",
    "Token.Generic.Error":      "#F92672",
    "Token.Generic.Heading":    "#A6E22E",
    "Token.Generic.Subheading": "#A6E22E",
    "Token.Generic.Deleted":    "#F92672",
    "Token.Generic.Inserted":   "#A6E22E",
    "Token.Generic.Output":     "#75715E",
    "Token.Generic.Prompt":     "#75715E",
    "Token.Generic.Emph":       "italic",
    "Token.Generic.Strong":     "bold",
    "Token.Text":               "#F8F8F2",
}

_DEFAULT_COLOR = "#F8F8F2"


def _resolve_color(token_type: str) -> str:
    """Walk up the token type hierarchy to find the closest monokai color."""
    ttype = str(token_type)
    while ttype:
        if ttype in _MONOKAI:
            val = _MONOKAI[ttype]
            if val.startswith("#"):
                return val
            break
        # Walk up: Token.Name.Function -> Token.Name -> Token
        parent = ttype.rsplit(".", 1)[0]
        if parent == ttype:
            # No more dots to strip; reached the root without a match
            break
        ttype = parent
    return _DEFAULT_COLOR


def _escape_ws(text: str, tab_width: int = 4) -> str:
    """XML-escape and make whitespace non-collapsing for ReportLab Paragraph.

    Paragraph parses its input as flowed markup (like HTML): leading whitespace
    is dropped and internal runs of spaces collapse to one. Escaping every
    space to `&nbsp;` after tab-expansion keeps indentation and alignment intact.
    """
    text = text.expandtabs(tab_width)
    return _html.escape(text).replace(" ", "&nbsp;")


def _normalize(code: str, tab_width: int = 4) -> str:
    """Expand tabs and dedent common leading indentation, once, up front."""
    return _textwrap.dedent(code.expandtabs(tab_width)).strip("\n")


def _autoformat(code: str, language: str | None, line_length: int = 72) -> str:
    """Best-effort black formatting. Returns code unchanged on any failure."""
    if (language or "").lower() not in ("python", "py", "python3"):
        return code
    try:
        import black
    except ImportError:
        return code
    try:
        return black.format_str(code, mode=black.Mode(line_length=line_length))
    except Exception:
        # syntax errors, partial snippets, non-parseable fragments
        return code


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def highlight_to_reportlab(code: str, language: str | None = None,
                            autoformat: bool = False, line_length: int = 72) -> str:
    """Highlight `code` and return ReportLab-XML string for a Paragraph.

    Returns a string of `<font color="...">token</font>` sequences joined with
    `<br/>` per line. Caller wraps in `Paragraph(xml, style)`.
    """
    if autoformat:
        code = _autoformat(code, language, line_length)
    code = _normalize(code)

    try:
        from pygments import lex
        from pygments.lexers import get_lexer_by_name, guess_lexer
        from pygments.util import ClassNotFound
    except ImportError:
        return _fallback_xml(code)

    if language:
        try:
            lexer = get_lexer_by_name(language, stripnl=False, stripall=False)
        except ClassNotFound:
            return _fallback_xml(code)
    else:
        try:
            lexer = guess_lexer(code)
        except Exception:
            return _fallback_xml(code)

    lines: list[str] = []
    current_line: list[str] = []

    for ttype, value in lex(code, lexer):
        # pygments may emit newlines as part of tokens
        parts = value.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                lines.append("".join(current_line))
                current_line = []
            if part:
                color = _resolve_color(ttype)
                escaped = _escape_ws(part)
                current_line.append(f'<font color="{color}">{escaped}</font>')

    if current_line:
        lines.append("".join(current_line))

    return "<br/>".join(lines) or "&nbsp;"


def highlight_to_table(code: str, language: str | None = None,
                       code_style=None, line_num_style=None,
                       autoformat: bool = False, line_length: int = 72) -> "Table":
    """Return a ReportLab Table with line numbers on the left and highlighted code on the right.

    Used by `add_code(highlight=True, line_numbers=True)`.
    """
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    if autoformat:
        code = _autoformat(code, language, line_length)
    code = _normalize(code)

    highlighted_xml = highlight_to_reportlab(code, language)
    highlighted_lines = highlighted_xml.split("<br/>")

    # Build table data: [line_num, code]
    data = []
    for num, line_xml in enumerate(highlighted_lines, 1):
        data.append([
            Paragraph(str(num), line_num_style),
            Paragraph(line_xml, code_style),
        ])

    table = Table(data, colWidths=[12 * mm, None])
    table.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (0, -1), 4),
        ("RIGHTPADDING", (0, 0), (0, -1), 4),
        ("LEFTPADDING", (1, 0), (1, -1), 6),
        ("RIGHTPADDING", (1, 0), (1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
    ]))
    return table


def _fallback_xml(code: str) -> str:
    """Plain escaped XML — same as v1 add_code, no highlighting."""
    lines = code.strip("\n").split("\n")
    escaped = [_escape_ws(ln) for ln in lines]
    return "<br/>".join(escaped) or "&nbsp;"


__all__ = ["highlight_to_reportlab", "highlight_to_table", "_escape_ws"]
