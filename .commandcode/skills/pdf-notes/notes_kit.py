"""notes_kit.py — Predefined rendering toolkit for PDF notes generation.

All helpers take `story` as first arg and mutate it in place.
Agent code should call these via `kit.add_*(story, ...)` — never import reportlab directly.

Usage:
    from notes_kit import *
    kit = NotesKit()
    # then in content.py:  kit.add_heading(story, "Title", level=1)
"""

from __future__ import annotations

import os
from typing import Callable

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, ListFlowable, ListItem, PageBreak, Image, KeepTogether,
    Frame, PageTemplate, BaseDocTemplate,
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------------------------
# 1. Font registration (Base14 default; optional custom fonts via env var)
# ---------------------------------------------------------------------------

_FONT_DIR = os.environ.get("PDF_SKILL_FONTS_DIR", "")

if _FONT_DIR:
    _inter_path = os.path.join(_FONT_DIR, "Inter-Regular.ttf")
    _inter_bold_path = os.path.join(_FONT_DIR, "Inter-Bold.ttf")
    _mono_path = os.path.join(_FONT_DIR, "JetBrainsMono-Regular.ttf")
    if os.path.isfile(_inter_path):
        pdfmetrics.registerFont(TTFont("Inter", _inter_path))
    if os.path.isfile(_inter_bold_path):
        pdfmetrics.registerFont(TTFont("Inter-Bold", _inter_bold_path))
    if os.path.isfile(_mono_path):
        pdfmetrics.registerFont(TTFont("JetBrainsMono", _mono_path))

# Use custom fonts if registered, else fall back to Base14
BODY_FONT = "Helvetica"
BODY_FONT_BOLD = "Helvetica-Bold"
MONO_FONT = "Courier"

try:
    from reportlab.pdfbase.pdfmetrics import getFont
    getFont("Inter")
    BODY_FONT = "Inter"
except Exception:
    pass

try:
    from reportlab.pdfbase.pdfmetrics import getFont
    getFont("Inter-Bold")
    BODY_FONT_BOLD = "Inter-Bold"
except Exception:
    pass

try:
    from reportlab.pdfbase.pdfmetrics import getFont
    getFont("JetBrainsMono")
    MONO_FONT = "JetBrainsMono"
except Exception:
    pass

# ---------------------------------------------------------------------------
# 2. Design tokens
# ---------------------------------------------------------------------------

INK = colors.HexColor("#1A1A1A")
HEADING = colors.HexColor("#0F172A")
ACCENT = colors.HexColor("#2563EB")
ACCENT_2 = colors.HexColor("#FF8A3D")
MUTED = colors.HexColor("#64748B")
CODE_BG = colors.HexColor("#0F172A")
CODE_TEXT = colors.HexColor("#E2E8F0")
TABLE_HEAD = colors.HexColor("#1E293B")
ROW_ALT = colors.HexColor("#F1F5F9")
BORDER = colors.HexColor("#CBD5E1")

CALLOUT_COLORS: dict[str, tuple[colors.HexColor, colors.HexColor]] = {
    "info":      (colors.HexColor("#EFF6FF"), colors.HexColor("#2563EB")),
    "tip":       (colors.HexColor("#ECFDF5"), colors.HexColor("#059669")),
    "warning":   (colors.HexColor("#FFFBEB"), colors.HexColor("#D97706")),
    "important": (colors.HexColor("#FEF2F2"), colors.HexColor("#DC2626")),
}

PAGE_SIZE = A4
MARGIN = 20 * mm

# v2 chart/diagram tokens
CHART_PALETTE = [
    "#2563EB",  # ACCENT  — series 1
    "#FF8A3D",  # ACCENT_2 — series 2
    "#059669",  # emerald — series 3
    "#D97706",  # amber  — series 4
    "#7C3AED",  # violet — series 5
    "#DC2626",  # red    — series 6
    "#0891B2",  # cyan   — series 7
    "#475569",  # slate  — series 8
]
CHART_DPI = 200
MATH_DPI = 300

# ---------------------------------------------------------------------------
# 3. Paragraph style sheet
# ---------------------------------------------------------------------------


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()

    # Remove conflicting built-in styles if present
    for name in ("Code", "Body", "Heading1", "Heading2", "Heading3"):
        if name in styles.byName:
            del styles.byName[name]
            styles.byAlias.pop(name.lower(), None)

    styles.add(ParagraphStyle(
        name="NotesTitle", fontName=BODY_FONT_BOLD, fontSize=26,
        textColor=HEADING, spaceAfter=4, leading=30,
    ))
    styles.add(ParagraphStyle(
        name="NotesSubtitle", fontName=BODY_FONT, fontSize=12,
        textColor=MUTED, spaceAfter=18, leading=16,
    ))
    styles.add(ParagraphStyle(
        name="H1", fontName=BODY_FONT_BOLD, fontSize=18,
        textColor=HEADING, spaceBefore=18, spaceAfter=8, leading=22,
        borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        name="H2", fontName=BODY_FONT_BOLD, fontSize=14,
        textColor=ACCENT, spaceBefore=14, spaceAfter=6, leading=18,
    ))
    styles.add(ParagraphStyle(
        name="H3", fontName=BODY_FONT_BOLD, fontSize=11.5,
        textColor=HEADING, spaceBefore=10, spaceAfter=4, leading=15,
    ))
    styles.add(ParagraphStyle(
        name="Body", fontName=BODY_FONT, fontSize=10.5,
        textColor=INK, leading=15, spaceAfter=8, alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name="BulletText", fontName=BODY_FONT, fontSize=10.5,
        textColor=INK, leading=14.5, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="Code", fontName=MONO_FONT, fontSize=9,
        textColor=CODE_TEXT, leading=13, backColor=None,
        wordWrap="CJK", splitLongWords=True,
    ))
    styles.add(ParagraphStyle(
        name="Caption", fontName=BODY_FONT, fontSize=10,
        textColor=MUTED, leading=13, alignment=TA_CENTER, spaceBefore=4,
    ))
    styles.add(ParagraphStyle(
        name="CalloutTitle", fontName=BODY_FONT_BOLD, fontSize=10,
        leading=13, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="CalloutBody", fontName=BODY_FONT, fontSize=9.5,
        textColor=INK, leading=13.5,
    ))
    styles.add(ParagraphStyle(
        name="Quote", fontName="Helvetica-Oblique", fontSize=10.5,
        textColor=MUTED, leading=15, leftIndent=14, spaceAfter=8,
    ))
    return styles


STYLES = build_styles()

# ---------------------------------------------------------------------------
# 4. Footer canvas (page numbers + running title)
# ---------------------------------------------------------------------------


class NotesCanvas(canvas.Canvas):
    """Adds page numbers + a running footer to every page."""

    def __init__(self, *args, doc_title: str = "Notes", **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_states: list[dict] = []
        self.doc_title = doc_title

    def showPage(self):
        self._saved_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_states)
        for i, state in enumerate(self._saved_states):
            self.__dict__.update(state)
            self._draw_footer(i + 1, total)
            super().showPage()
        super().save()

    def _draw_footer(self, page_num: int, total_pages: int):
        self.setFont(BODY_FONT, 8)
        self.setFillColor(MUTED)
        self.drawString(MARGIN, 12 * mm, self.doc_title)
        self.drawRightString(
            PAGE_SIZE[0] - MARGIN, 12 * mm, f"Page {page_num} of {total_pages}"
        )
        self.setStrokeColor(BORDER)
        self.line(MARGIN, 16 * mm, PAGE_SIZE[0] - MARGIN, 16 * mm)


# ---------------------------------------------------------------------------
# 5. Helper functions (all mutate `story` in place)
# ---------------------------------------------------------------------------


def add_title_block(story: list, title: str, subtitle: str | None = None):
    """Append a title block with optional subtitle and a rule."""
    story.append(Paragraph(_normalize_unicode_math(title), STYLES["NotesTitle"]))
    if subtitle:
        story.append(Paragraph(_normalize_unicode_math(subtitle), STYLES["NotesSubtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceAfter=14))


def add_heading(story: list, text: str, level: int = 1):
    """Append H1 / H2 / H3 heading. H1 gets a thin rule under it."""
    style = {1: "H1", 2: "H2", 3: "H3"}.get(level, "H1")
    story.append(Paragraph(_normalize_unicode_math(str(text)), STYLES[style]))
    if level == 1:
        story.append(HRFlowable(width="100%", thickness=0.6, color=BORDER, spaceAfter=6))


_UNICODE_SUBS = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "ₐ": "a", "₆": "6", "ₑ": "e", "ₕ": "h", "ᵢ": "i",
    "ⱼ": "j", "ₖ": "k", "ₗ": "l", "ₘ": "m", "ₙ": "n",
    "ₒ": "o", "ₚ": "p", "ᵣ": "r", "ₛ": "s", "ₜ": "t",
    "ᵤ": "u", "ᵥ": "v", "ₓ": "x",
    "₊": "+", "₋": "-", "₌": "=", "₍": "(", "₎": ")",
}
_UNICODE_SUPERS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁼": "=", "⁽": "(", "⁾": ")",
    "ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e",
    "ᶠ": "f", "ᵍ": "g", "ʰ": "h", "ⁱ": "i", "ʲ": "j",
    "ᵏ": "k", "ˡ": "l", "ᵐ": "m", "ⁿ": "n", "ᵒ": "o",
    "ᵖ": "p", "ʳ": "r", "ˢ": "s", "ᵗ": "t", "ᵘ": "u",
    "ᵛ": "v", "ʷ": "w", "ˣ": "x", "ʸ": "y", "ᶻ": "z",
}


def _normalize_unicode_math(text: str) -> str:
    """Replace Unicode subscript/superscript glyphs with ReportLab <sub>/<super> tags.

    Base14 PDF fonts (Helvetica, Courier) lack subscript/superscript code points,
    so characters like ₀ₕᵢ₀²² render as black squares (■). The skill convention
    (see SKILL.md) is to use <sub>/<super> tags or add_math instead. This helper
    auto-converts the most common ones so chapter code that slips in Unicode
    renders correctly rather than corrupted.

    Only acts on bare Unicode glyphs; XML-tagged content (<sub>, <super>) is left
    untouched since it doesn't contain those Unicode code points.
    """
    if not text:
        return text
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _UNICODE_SUBS:
            # Greedy: collect run of subscripts into one <sub> tag
            j = i
            buf = []
            while j < len(text) and text[j] in _UNICODE_SUBS:
                buf.append(_UNICODE_SUBS[text[j]])
                j += 1
            out.append(f"<sub>{''.join(buf)}</sub>")
            i = j
        elif ch in _UNICODE_SUPERS:
            j = i
            buf = []
            while j < len(text) and text[j] in _UNICODE_SUPERS:
                buf.append(_UNICODE_SUPERS[text[j]])
                j += 1
            out.append(f"<super>{''.join(buf)}</super>")
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def add_paragraph(story: list, text: str):
    """Append a justified body paragraph. Supports inline XML tags.

    Unicode subscript/superscript glyphs (₀ ² ₕ etc.) are auto-converted to
    <sub>/<super> tags — Base14 fonts lack those code points.
    """
    story.append(Paragraph(_normalize_unicode_math(str(text)), STYLES["Body"]))


def add_bullets(story: list, items: list[str], bullet_char: str = "\u2022"):
    """Append a bulleted list."""
    flowables = [
        ListItem(Paragraph(_normalize_unicode_math(str(item)), STYLES["BulletText"]), spaceAfter=3)
        for item in items
    ]
    story.append(ListFlowable(
        flowables, bulletType="bullet", bulletChar=bullet_char,
        leftIndent=16, bulletFontSize=8, bulletColor=ACCENT,
    ))
    story.append(Spacer(1, 6))


def add_numbered_list(story: list, items: list[str]):
    """Append a numbered list."""
    flowables = [
        ListItem(Paragraph(_normalize_unicode_math(str(item)), STYLES["BulletText"]), spaceAfter=3)
        for item in items
    ]
    story.append(ListFlowable(
        flowables, bulletType="1", leftIndent=16,
        bulletFontSize=9.5, bulletColor=ACCENT,
    ))
    story.append(Spacer(1, 6))


def add_code(story: list, code: str, language: str | None = None,
             highlight: bool = False, line_numbers: bool = False,
             caption: str | None = None, autoformat: bool = False,
             line_length: int = 72):
    """Append a dark-background code block. highlight=True enables syntax coloring.

    autoformat=True runs `black` over Python snippets before highlighting
    (best-effort; unparseable/partial snippets are left unchanged).
    """
    if highlight:
        try:
            from highlighting import highlight_to_reportlab, highlight_to_table
            code_style = ParagraphStyle(
                "HighlightedCode", fontName=MONO_FONT, fontSize=9,
                textColor=CODE_TEXT, leading=13,
                wordWrap="CJK", splitLongWords=True,
            )
            line_num_style = ParagraphStyle(
                "LineNum", fontName=MONO_FONT, fontSize=9,
                textColor=MUTED, leading=13,
            )
            if line_numbers:
                inner = highlight_to_table(code, language, code_style, line_num_style,
                                            autoformat=autoformat, line_length=line_length)
            else:
                xml = highlight_to_reportlab(code, language,
                                              autoformat=autoformat, line_length=line_length)
                inner = Paragraph(xml, code_style)

            wrapper = Table(
                [[inner]],
                colWidths=[PAGE_SIZE[0] - 2 * MARGIN],
            )
            wrapper.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("ROUNDEDCORNERS", [4, 4, 4, 4]),
            ]))
            if language:
                story.append(Paragraph(language.upper(), ParagraphStyle(
                    name="LangTagHL", fontName=BODY_FONT_BOLD, fontSize=7.5,
                    textColor=MUTED, spaceAfter=2,
                )))
            story.append(wrapper)
            story.append(Spacer(1, 8))
            if caption:
                story.append(Paragraph(caption, STYLES["Caption"]))
            return
        except Exception:
            pass  # fall through to plain code

    # Plain (v1) code block — no syntax coloring
    from highlighting import _escape_ws, _normalize
    rows = _normalize(code).split("\n")
    escaped = [_escape_ws(ln) for ln in rows]
    body = "<br/>".join(escaped) or "&nbsp;"
    table = Table(
        [[Paragraph(body, STYLES["Code"])]],
        colWidths=[PAGE_SIZE[0] - 2 * MARGIN],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    if language:
        story.append(Paragraph(language.upper(), ParagraphStyle(
            name="LangTag", fontName=BODY_FONT_BOLD, fontSize=7.5,
            textColor=MUTED, spaceAfter=2,
        )))
    story.append(table)
    story.append(Spacer(1, 8))
    if caption:
        story.append(Paragraph(caption, STYLES["Caption"]))


def add_callout(story: list, text: str, kind: str = "info", title: str | None = None):
    """Append a colored callout/note box (info, tip, warning, important)."""
    bg, border = CALLOUT_COLORS.get(kind, CALLOUT_COLORS["info"])
    default_titles = {
        "info": "Note", "tip": "Tip", "warning": "Warning", "important": "Important"
    }
    title = title or default_titles.get(kind, "Note")

    title_style = ParagraphStyle(
        "CalloutTitleColored", parent=STYLES["CalloutTitle"], textColor=border,
    )
    content = [
        Paragraph(_normalize_unicode_math(title), title_style),
        Paragraph(_normalize_unicode_math(str(text)), STYLES["CalloutBody"]),
    ]
    table = Table([[content]], colWidths=[PAGE_SIZE[0] - 2 * MARGIN])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 1, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))


def add_table(story: list, headers: list[str], rows: list[list[str]], col_widths: list | None = None):
    """Append a table with header row + zebra striping."""
    data = [[Paragraph(f"<b>{h}</b>", ParagraphStyle(
        "TH", fontName=BODY_FONT_BOLD, fontSize=9.5, textColor=colors.white))
        for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), STYLES["BulletText"]) for c in row])

    table = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEAD),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))
    table.setStyle(TableStyle(style))
    story.append(table)
    story.append(Spacer(1, 10))


def add_divider(story: list):
    """Append a thin horizontal hairline."""
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 8))


def add_space(story: list, height: float = 8):
    """Append blank vertical space (in points)."""
    story.append(Spacer(1, height))


def add_page_break(story: list):
    """Force the next content onto a new page."""
    story.append(PageBreak())


def add_quote(story: list, text: str, attribution: str | None = None):
    """Append a styled blockquote, optionally with attribution."""
    txt = _normalize_unicode_math(text)
    if attribution:
        txt = f'{txt} <i>\u2014 {_normalize_unicode_math(attribution)}</i>'
    story.append(Paragraph(txt, STYLES["Quote"]))


def _resolve_image_path(path: str) -> str:
    """Resolve an image path, trying CWD then graphics_kit's run_dir for relative paths.

    Agents commonly write ``kit.add_image(story, "assets/page_03.png")`` where
    ``assets/`` lives in the current run directory. CWD may or may not be set
    to the run dir at call time (depends on runner's chdir), so we fall back
    to ``graphics_kit.get_run_dir()`` when the path is not found directly.
    """
    if os.path.isfile(path):
        return path
    # Try resolving relative paths against the graphics_kit run dir
    if not os.path.isabs(path):
        try:
            from graphics_kit import get_run_dir
            run_dir = get_run_dir()
            if run_dir:
                candidate = os.path.join(run_dir, path)
                if os.path.isfile(candidate):
                    return candidate
        except Exception:
            pass
    return path  # original path; add_image will show the missing-image placeholder


def add_image(story: list, path: str, width: float | None = None, caption: str | None = None):
    """Append an image with optional fixed width and caption. Graceful on missing file."""
    resolved = _resolve_image_path(path)
    if not os.path.isfile(resolved):
        # Red-bordered placeholder for missing image
        placeholder_style = ParagraphStyle(
            "MissingImage", fontName=BODY_FONT, fontSize=9,
            textColor=colors.HexColor("#DC2626"), alignment=TA_CENTER,
        )
        story.append(Paragraph(f"[missing image: {path}]", placeholder_style))
        story.append(Spacer(1, 8))
        return

    img = Image(resolved)
    if width:
        ratio = width / img.imageWidth
        img.drawWidth = width
        img.drawHeight = img.imageHeight * ratio
    story.append(img)
    if caption:
        story.append(Paragraph(_normalize_unicode_math(caption), STYLES["Caption"]))
        # Small spacer after caption - tight layout for figures/equations
        story.append(Spacer(1, 4))
    else:
        # Even tighter when uncaptioned (e.g., inline math, small formulas)
        story.append(Spacer(1, 3))


def add_file_tree(story: list, spec: dict, caption: str | None = None):
    """Append an icon-based file/folder tree diagram with auto-pagination.

    spec: nested dict with node_type (drive|folder|file|ellipsis), label, children.
    Splits across pages if the tree row count exceeds one page.
    """
    validation_error = validate_file_tree_spec(spec)
    if validation_error:
        story.append(Paragraph(validation_error, ParagraphStyle(
            "TreeError", fontName=BODY_FONT, fontSize=9,
            textColor=colors.HexColor("#DC2626"), leading=13,
        )))
        story.append(Spacer(1, 8))
        return

    available_width = PAGE_SIZE[0] - 2 * MARGIN
    usable_height = PAGE_SIZE[1] - 2 * MARGIN - 8 * mm

    from file_tree import _count_rows, split_tree_spec

    total_rows = _count_rows(spec)
    max_rows = int((usable_height - FT_VERT_PAD * 2) / FT_ROW_HEIGHT)
    max_rows = max(max_rows, 1)

    blocks = split_tree_spec(spec, max_rows) if total_rows > max_rows else [spec]

    if caption:
        story.append(Paragraph(caption, STYLES["Caption"]))
    for block in blocks:
        story.append(IconTreeBlock(block, width=available_width))
    story.append(Spacer(1, 8))


def add_keep_together(story: list, *flowable_builders: Callable[[list], None]):
    """Wrap multiple flowable builders into a KeepTogether block to prevent page splits."""
    temp: list = []
    for build in flowable_builders:
        build(temp)
    story.append(KeepTogether(temp))


def render_latex_to_png(expr: str, path: str, fontsize: int = 16):
    """Render a LaTeX expression to a PNG file via matplotlib (lazy import)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for render_latex_to_png. "
            "Install it with: uv add matplotlib"
        ) from e

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${expr}$", fontsize=fontsize)
    fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True, pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Top-level assembly
# ---------------------------------------------------------------------------


def generate_notes_pdf(
    output_path: str,
    title: str,
    subtitle: str | None,
    build_content_fn: Callable[[list], None],
    theme: str = "light",
):
    """Build a complete PDF from a content-builder function."""
    doc = SimpleDocTemplate(
        output_path, pagesize=PAGE_SIZE,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN + 8 * mm,
        title=title,
    )

    story: list = []
    build_content_fn(story)

    doc.build(
        story,
        canvasmaker=lambda *a, **kw: NotesCanvas(*a, doc_title=title, **kw),
    )


# ---------------------------------------------------------------------------
# 7. Re-export graphics + highlighting helpers
# ---------------------------------------------------------------------------

from graphics_kit import (  # noqa: E402, F401
    add_bar_chart, add_line_chart, add_pie_chart, add_donut_chart,
    add_scatter, add_histogram, add_box_plot, add_area_chart,
    add_heatmap, add_radar_chart,
    add_flowchart, add_tree, add_state_diagram, add_block_diagram,
    add_graph, add_call_graph, add_venn, add_timeline,
    add_math, add_image_rasterize,
    get_graphics_warnings, set_run_dir, get_run_dir,
)

from file_tree import (  # noqa: E402, F401
    IconTreeBlock, validate_file_tree_spec, split_tree_spec,
    FOLDER_FILL as FT_FOLDER_FILL, FOLDER_LINE as FT_FOLDER_LINE,
    FILE_FILL as FT_FILE_FILL, FILE_LINE as FT_FILE_LINE,
    DRIVE_COLOR as FT_DRIVE_COLOR,
    ROW_HEIGHT as FT_ROW_HEIGHT, VERT_PAD as FT_VERT_PAD,
)

# highlighting.py is imported lazily inside add_code when highlight=True
# but we also re-export the public function for direct use
from highlighting import highlight_to_reportlab, highlight_to_table  # noqa: E402, F401


# ---------------------------------------------------------------------------
# 8. Export everything
# ---------------------------------------------------------------------------

__all__ = [
    # Design tokens
    "INK", "HEADING", "ACCENT", "ACCENT_2", "MUTED",
    "CODE_BG", "CODE_TEXT", "TABLE_HEAD", "ROW_ALT", "BORDER",
    "CALLOUT_COLORS", "PAGE_SIZE", "MARGIN",
    # Units (from reportlab.lib.units)
    "mm", "inch",
    # Colors (from reportlab.lib.colors)
    "colors",
    # Chart palette
    "CHART_PALETTE", "CHART_DPI", "MATH_DPI",
    # Font names
    "BODY_FONT", "BODY_FONT_BOLD", "MONO_FONT",
    # Styles
    "STYLES", "build_styles",
    # Canvas
    "NotesCanvas",
    # Text helpers
    "add_title_block", "add_heading", "add_paragraph",
    "add_bullets", "add_numbered_list", "add_code",
    "add_callout", "add_table", "add_divider",
    "add_space", "add_page_break", "add_quote",
    "add_image", "add_keep_together",
    # File tree
    "add_file_tree", "IconTreeBlock",
    "validate_file_tree_spec", "split_tree_spec",
    # Extensions
    "render_latex_to_png",
    # Charts (from graphics_kit)
    "add_bar_chart", "add_line_chart", "add_pie_chart", "add_donut_chart",
    "add_scatter", "add_histogram", "add_box_plot", "add_area_chart",
    "add_heatmap",
    # Diagrams (from graphics_kit)
    "add_flowchart", "add_tree", "add_state_diagram", "add_block_diagram",
    "add_graph", "add_call_graph", "add_venn", "add_timeline",
    # Math (from graphics_kit)
    "add_math",
    # Rasterized external source import (SVG / PDF page -> cached PNG)
    "add_image_rasterize",
    # Highlighting (from highlighting)
    "highlight_to_reportlab", "highlight_to_table",
    # Warnings
    "get_graphics_warnings", "set_run_dir", "get_run_dir",
    # Assembly
    "generate_notes_pdf",
]
