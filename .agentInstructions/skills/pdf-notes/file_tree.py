"""file_tree.py — Icon-based file tree diagram Flowable for PDF notes.

Pure vector drawing via ReportLab canvas primitives — no external image/SVG deps.
Renders drive/folder/file nodes with tree connector lines and arrowheads.
"""

from reportlab.platypus import Flowable
from reportlab.lib.colors import HexColor

ALLOWED_NODE_TYPES = {"drive", "folder", "file", "ellipsis"}

FOLDER_FILL = HexColor("#F5D06B")
FOLDER_LINE = HexColor("#8a6d1f")
FILE_FILL = HexColor("#E8ECF2")
FILE_LINE = HexColor("#5a6472")
DRIVE_COLOR = HexColor("#2b2b2b")
INK = HexColor("#1a1a1a")

FONT = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_SIZE = 11

ICON_WIDTH = 20
ROW_HEIGHT = 24
VERT_PAD = 16


# -----------------------------------------------------------------------
# Icon drawing functions
# -----------------------------------------------------------------------

def _draw_drive_icon(c, x, y, size=12):
    c.setStrokeColor(DRIVE_COLOR)
    c.setLineWidth(1.3)
    c.line(x, y + size * 0.75, x + size, y + size * 0.75)
    c.line(x, y + size * 0.45, x + size, y + size * 0.45)
    c.line(x, y + size * 0.15, x + size, y + size * 0.15)


def _draw_folder_icon(c, x, y, size=12):
    w, h = size * 1.3, size * 0.9
    tab_w = w * 0.45
    c.setFillColor(FOLDER_FILL)
    c.setStrokeColor(FOLDER_LINE)
    c.setLineWidth(0.8)
    c.rect(x, y + h * 0.65, tab_w, h * 0.35, fill=1, stroke=1)
    c.rect(x, y, w, h * 0.75, fill=1, stroke=1)


def _draw_file_icon(c, x, y, size=12):
    w, h = size * 0.9, size * 1.15
    c.setFillColor(FILE_FILL)
    c.setStrokeColor(FILE_LINE)
    c.setLineWidth(0.8)
    c.rect(x, y, w, h, fill=1, stroke=1)
    c.setStrokeColor(FILE_LINE)
    c.setLineWidth(0.6)
    for i in range(3):
        ly = y + h * 0.25 + i * h * 0.2
        c.line(x + w * 0.15, ly, x + w * 0.85, ly)


ICON_DRAW = {
    "drive": _draw_drive_icon,
    "folder": _draw_folder_icon,
    "file": _draw_file_icon,
}


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------

def validate_file_tree_spec(spec: dict) -> str | None:
    """Validate a file_tree content block spec. Returns error string or None."""

    def _validate_node(node, path="root"):
        node_type = node.get("node_type")
        label = node.get("label")
        children = node.get("children", [])

        if node_type not in ALLOWED_NODE_TYPES:
            return f"Invalid node_type '{node_type}' at {path} — must be one of: {', '.join(sorted(ALLOWED_NODE_TYPES))}"
        if node_type != "ellipsis" and not label:
            return f"Missing required 'label' for node_type '{node_type}' at {path}"
        if not isinstance(children, list):
            return f"'children' must be a list at {path}"

        for i, child in enumerate(children):
            err = _validate_node(child, f"{path}.children[{i}]")
            if err:
                return err
        return None

    if not isinstance(spec, dict):
        return "file_tree spec must be a dict"
    return _validate_node(spec)


# -----------------------------------------------------------------------
# Tree row counting & flattening
# -----------------------------------------------------------------------

def _count_rows(spec: dict) -> int:
    count = 1
    for child in spec.get("children", []):
        count += _count_rows(child)
    return count


def _flatten(spec: dict, rows: list, depth: int, ancestor_has_more: list[bool]):
    rows.append((depth, spec, list(ancestor_has_more)))
    children = spec.get("children", [])
    for i, child in enumerate(children):
        has_more = i < len(children) - 1
        _flatten(child, rows, depth + 1, ancestor_has_more + [has_more])


# -----------------------------------------------------------------------
# Tree splitting for pagination
# -----------------------------------------------------------------------

def _collect_blocks(spec: dict, max_rows: int, blocks: list[dict]):
    """Recursively collect subtrees that each fit within max_rows."""
    total = _count_rows(spec)
    if total <= max_rows:
        blocks.append(spec)
        return

    children = spec.get("children", [])
    remaining = list(children)

    while remaining:
        group = []
        group_rows = 1
        while remaining:
            child_rows = _count_rows(remaining[0])
            if group_rows + child_rows > max_rows:
                if not group:
                    _collect_blocks(remaining[0], max_rows, blocks)
                    remaining.pop(0)
                break
            group.append(remaining.pop(0))
            group_rows += child_rows

        if group:
            blocks.append(dict(spec, children=group))


def split_tree_spec(spec: dict, max_rows_per_block: int) -> list[dict]:
    """Split a tree spec into multiple specs that each fit within max_rows.

    Never splits a node from its own children mid-branch — safe boundaries only.
    """
    if max_rows_per_block <= 0:
        return [spec]
    blocks: list[dict] = []
    _collect_blocks(spec, max_rows_per_block, blocks)
    if not blocks:
        blocks = [spec]
    return blocks


# -----------------------------------------------------------------------
# IconTreeBlock Flowable
# -----------------------------------------------------------------------

class IconTreeBlock(Flowable):
    """ReportLab Flowable that renders a file/folder tree as icon-based vector diagram.

    spec: nested dict with node_type, label, children
    """

    def __init__(self, spec: dict, width: float = 420, indent: float = 28):
        super().__init__()
        self.spec = spec
        self._width = width
        self.indent = indent
        self.rows: list[tuple[int, dict, list[bool]]] = []
        _flatten(spec, self.rows, 0, [])
        self.height = VERT_PAD * 2 + ROW_HEIGHT * len(self.rows)

    def wrap(self, availWidth, availHeight):
        self._width = availWidth
        return (self._width, self.height)

    def draw(self):
        c = self.canv
        c.saveState()

        top_y = self.height - VERT_PAD
        row_h = ROW_HEIGHT
        row_y = [top_y - row_h * (i + 0.5) for i in range(len(self.rows))]

        for i, (depth, node, ancestors) in enumerate(self.rows):
            y = row_y[i]
            base_x = VERT_PAD

            for level, has_more in enumerate(ancestors[:-1] if ancestors else []):
                if has_more:
                    lx = base_x + level * self.indent + 4
                    c.setStrokeColor(INK)
                    c.setLineWidth(1)
                    c.line(lx, y - row_h / 2, lx, y + row_h / 2)

            if depth > 0:
                level = depth - 1
                lx = base_x + level * self.indent + 4
                is_last = not ancestors[-1] if ancestors else True
                top_seg = y + row_h / 2
                bottom_seg = y if is_last else y - row_h / 2
                c.setStrokeColor(INK)
                c.setLineWidth(1)
                c.line(lx, top_seg, lx, bottom_seg)
                arrow_x1 = base_x + depth * self.indent
                c.line(lx, y, arrow_x1, y)
                c.line(arrow_x1, y, arrow_x1 - 4, y + 3)
                c.line(arrow_x1, y, arrow_x1 - 4, y - 3)

            icon_x = base_x + depth * self.indent + 2
            label_x = icon_x + ICON_WIDTH

            node_type = node.get("node_type", "folder")
            if node_type == "ellipsis":
                c.setFillColor(INK)
                c.setFont(FONT, FONT_SIZE)
                c.drawString(icon_x, y - 5, "...")
                continue

            draw_fn = ICON_DRAW.get(node_type, _draw_folder_icon)
            draw_fn(c, icon_x, y - 6)

            font = FONT_BOLD if node_type == "drive" else FONT
            c.setFillColor(INK)
            c.setFont(font, FONT_SIZE)
            c.drawString(label_x, y - 4, node.get("label", ""))

        c.restoreState()
