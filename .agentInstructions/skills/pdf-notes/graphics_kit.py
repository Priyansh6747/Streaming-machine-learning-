"""graphics_kit.py — Charts, diagrams, math, and visual helpers for PDF notes.

All helpers take `story` as first arg and mutate it in place.
Imported by notes_kit.py and re-exported so the agent sees a single `kit.*` surface.

Charts use matplotlib (pure python, no system deps).
Diagrams use the graphviz Python wrapper (requires system `dot` binary).
Math uses matplotlib mathtext.
All degrade gracefully on bad input or missing deps.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from reportlab.lib.units import mm

_CHART_DEFAULT_W = 170 * mm  # full content-frame width on A4 with 20mm margins

# ---------------------------------------------------------------------------
# Lazy imports — matplotlib, graphviz, numpy, PIL
# These are imported at function call time (not module level) so the module
# loads cleanly even if a dep is missing.
# ---------------------------------------------------------------------------


def _import_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _import_graphviz():
    import graphviz
    return graphviz


def _import_numpy():
    import numpy as np
    return np


def _import_pil():
    from PIL import Image
    return Image


# ---------------------------------------------------------------------------
# Theme injection — read tokens from notes_kit so charts stay on-brand
# ---------------------------------------------------------------------------

_CHART_PALETTE = [
    "#2563EB",  # ACCENT  — series 1
    "#FF8A3D",  # ACCENT_2 — series 2
    "#059669",  # emerald — series 3
    "#D97706",  # amber  — series 4
    "#7C3AED",  # violet — series 5
    "#DC2626",  # red    — series 6
    "#0891B2",  # cyan   — series 7
    "#475569",  # slate  — series 8
]

_CHART_DPI = 200
_MATH_DPI = 300
_CODE_BG = "#0F172A"
_MUTED = "#64748B"
_BORDER = "#CBD5E1"
_INK = "#1A1A1A"
_HEADING = "#0F172A"

# Module-level warnings accumulator (collected by runner after build)
_warnings: list[str] = []

# Module-level run directory (set by runner before build, read by helpers)
_current_run_dir: str | None = None


def _warn(msg: str):
    _warnings.append(msg)


def get_graphics_warnings() -> list[str]:
    """Return and clear the warnings list. Called by runner after build."""
    global _warnings
    out = list(_warnings)
    _warnings = []
    return out


def set_run_dir(path: str):
    """Set the current run directory for asset storage. Called by runner."""
    global _current_run_dir
    _current_run_dir = path


def get_run_dir() -> str | None:
    """Get the current run directory."""
    return _current_run_dir


# ---------------------------------------------------------------------------
# Asset cache
# ---------------------------------------------------------------------------

def _asset_hash(helper_name: str, args: tuple) -> str:
    """Stable short hash of a helper call for cache filenames."""
    h = hashlib.sha1(repr((helper_name, args)).encode()).hexdigest()[:8]
    return f"{helper_name}_{h}.png"


def _ensure_assets_dir(run_dir: Path) -> Path:
    assets = run_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    return assets


def _run_dir_from_story(story: list) -> Path:
    """Get the run directory. Falls back to CWD/workspace."""
    if _current_run_dir:
        return Path(_current_run_dir)
    return Path.cwd() / "workspace" / "_tmp"


def _setup_mpl_theme():
    """Apply brand theme to matplotlib rcParams. Called once per chart render."""
    plt = _import_mpl()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans"],
        "text.color": _INK,
        "axes.labelcolor": _INK,
        "axes.edgecolor": _BORDER,
        "xtick.color": _MUTED,
        "ytick.color": _MUTED,
        "axes.grid": True,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
    })
    return plt


def _embed_asset(story: list, asset_path: str, width: float | None = None,
                 caption: str | None = None, default_width: float | None = None):
    """Embed a rendered asset PNG into the story via notes_kit.add_image.

    Width/height resolution (so the image never overflows the content frame):
      1. Explicit `width` argument (always honored, but capped to frame width).
      2. `default_width` (used for charts/diagrams so high-DPI PNGs don't
         render at natural pixel size).
      3. Natural image width, capped to the frame width (used for math,
         so small formulas stay small but large ones never overflow).
      4. After width is resolved, if the resulting height exceeds the max
         content height (~680pt on A4), the width is scaled down proportionally.
    """
    from notes_kit import add_image
    frame_w = _CHART_DEFAULT_W  # content-frame width on A4 with 20mm margins
    max_h = 680  # max content height in points (A4 height minus margins minus footer)
    if width is not None:
        eff = min(width, frame_w)
    elif default_width is not None:
        eff = min(default_width, frame_w)
    else:
        # Auto-fit (math): use natural width, but cap to frame if larger
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(asset_path) as im:
                natural = im.width
        except Exception:
            natural = frame_w
        eff = min(natural, frame_w)
    # Height cap: if the image at `eff` width would exceed max_h, scale down
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(asset_path) as im:
            img_w, img_h = im.size
        if img_w > 0:
            ratio = eff / img_w
            rendered_h = img_h * ratio
            if rendered_h > max_h:
                eff = eff * (max_h / rendered_h)
    except Exception:
        pass
    add_image(story, asset_path, width=eff, caption=caption)


def _check_dot_binary() -> bool:
    """Check if graphviz 'dot' binary is available."""
    return shutil.which("dot") is not None


# ===========================================================================
# CHART HELPERS (matplotlib)
# ===========================================================================


def _render_chart_to_png(fig, path: str, dpi: int = _CHART_DPI):
    """Save a matplotlib figure to PNG with transparent background."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True, pad_inches=0.15)
    import matplotlib.pyplot as plt
    plt.close(fig)


def add_bar_chart(story: list, title: str | None = None,
                  categories: list[str] | None = None,
                  series: list[dict] | None = None,
                  values: list[float] | None = None,
                  xlabel: str | None = None, ylabel: str | None = None,
                  horizontal: bool = False,
                  caption: str | None = None, width: float | None = None):
    """Append a grouped bar chart. Single-series via `values=`, multi-series via `series=`."""
    try:
        plt = _setup_mpl_theme()
        categories = categories or []

        # Normalize series
        if series is None and values is not None:
            series = [{"label": "", "values": values}]
        if not series:
            _warn("add_bar_chart: no data provided")
            series = [{"label": "", "values": [0]}]
            if not categories:
                categories = [""]

        # Coerce lengths
        max_len = max(len(s.get("values", [])) for s in series)
        if not categories:
            categories = [str(i) for i in range(max_len)]
        min_len = min(len(categories), max_len)
        categories = categories[:min_len]
        for s in series:
            vals = s.get("values", [])
            s["values"] = (vals + [0] * max_len)[:min_len]

        import numpy as np
        x = np.arange(len(categories))
        n = len(series)
        bar_width = 0.8 / n

        fig, ax = plt.subplots(figsize=(7, 4))
        for i, s in enumerate(series):
            color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
            offset = x - 0.4 + bar_width * (i + 0.5)
            if horizontal:
                ax.barh(offset, s["values"], bar_width, label=s.get("label", ""),
                        color=color, edgecolor="none")
            else:
                ax.bar(offset, s["values"], bar_width, label=s.get("label", ""),
                       color=color, edgecolor="none")

        if horizontal:
            ax.set_yticks(x)
            ax.set_yticklabels(categories)
            if xlabel:
                ax.set_xlabel(xlabel)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            if xlabel:
                ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", color=_HEADING, pad=10)
        if any(s.get("label") for s in series):
            ax.legend(frameon=False, fontsize=9)

        # Cache + embed
        run_dir = _run_dir_from_story(story)
        assets_dir = _ensure_assets_dir(run_dir)
        fname = _asset_hash("bar_chart", (title, categories, str(series), horizontal))
        path = str(assets_dir / fname)
        if not os.path.isfile(path):
            _render_chart_to_png(fig, path)
        _embed_asset(story, path, width=width, caption=caption, default_width=_CHART_DEFAULT_W)
    except Exception as e:
        _warn(f"add_bar_chart failed: {e}")
        from notes_kit import add_paragraph
        add_paragraph(story, f"[chart render failed: {e}]")


def add_line_chart(story: list, title: str | None = None,
                   series: list[dict] | None = None,
                   xlabel: str | None = None, ylabel: str | None = None,
                   markers: bool = True,
                   caption: str | None = None, width: float | None = None):
    """Append a multi-series line chart."""
    try:
        plt = _setup_mpl_theme()
        series = series or []
        if not series:
            _warn("add_line_chart: no series provided")
            return

        linestyles = ["-", "--", ":", "-."]
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, s in enumerate(series):
            color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
            ls = linestyles[i % len(linestyles)]
            marker = "o" if markers else None
            xs = s.get("xs", list(range(len(s.get("ys", [])))))
            ys = s.get("ys", [])
            ax.plot(xs, ys, color=color, linestyle=ls, marker=marker,
                    markersize=4, linewidth=2, label=s.get("label", ""))

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", color=_HEADING, pad=10)
        if any(s.get("label") for s in series):
            ax.legend(frameon=False, fontsize=9)

        run_dir = _run_dir_from_story(story)
        assets_dir = _ensure_assets_dir(run_dir)
        fname = _asset_hash("line_chart", (title, str(series)))
        path = str(assets_dir / fname)
        if not os.path.isfile(path):
            _render_chart_to_png(fig, path)
        _embed_asset(story, path, width=width, caption=caption, default_width=_CHART_DEFAULT_W)
    except Exception as e:
        _warn(f"add_line_chart failed: {e}")
        from notes_kit import add_paragraph
        add_paragraph(story, f"[chart render failed: {e}]")


def add_pie_chart(story: list, title: str | None = None,
                  labels: list[str] | None = None,
                  values: list[float] | None = None,
                  donut: bool = False,
                  caption: str | None = None, width: float | None = None):
    """Append a pie or donut chart."""
    try:
        plt = _setup_mpl_theme()
        labels = labels or []
        values = values or []
        if not values:
            _warn("add_pie_chart: no values provided")
            return
        # Coerce negatives
        values = [max(0, v) for v in values]
        min_len = min(len(labels), len(values))
        labels = labels[:min_len] or [f"Slice {i+1}" for i in range(min_len)]
        values = values[:min_len]
        colors = [_CHART_PALETTE[i % len(_CHART_PALETTE)] for i in range(len(values))]

        fig, ax = plt.subplots(figsize=(5, 5))
        wedge_kwargs = {"width": 0.4} if donut else {}
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors, autopct="%1.0f%%",
            startangle=90, pctdistance=0.75 if donut else 0.6,
            wedgeprops=wedge_kwargs if donut else None,
            textprops={"fontsize": 9},
        )
        for t in autotexts:
            t.set_fontsize(8)
            t.set_color("white")
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", color=_HEADING, pad=10)

        run_dir = _run_dir_from_story(story)
        assets_dir = _ensure_assets_dir(run_dir)
        fname = _asset_hash("pie_chart", (title, labels, values, donut))
        path = str(assets_dir / fname)
        if not os.path.isfile(path):
            _render_chart_to_png(fig, path)
        _embed_asset(story, path, width=width, caption=caption, default_width=_CHART_DEFAULT_W)
    except Exception as e:
        _warn(f"add_pie_chart failed: {e}")
        from notes_kit import add_paragraph
        add_paragraph(story, f"[chart render failed: {e}]")


def add_donut_chart(story: list, **kwargs):
    """Shorthand for add_pie_chart(..., donut=True)."""
    add_pie_chart(story, donut=True, **kwargs)


def add_scatter(story: list, title: str | None = None,
                xs: list[float] | None = None, ys: list[float] | None = None,
                labels: list[str] | None = None,
                xlabel: str | None = None, ylabel: str | None = None,
                trendline: bool = False,
                caption: str | None = None, width: float | None = None):
    """Append a scatter plot, optionally with a linear trendline."""
    try:
        plt = _setup_mpl_theme()
        xs = xs or []
        ys = ys or []
        if not xs or not ys:
            _warn("add_scatter: no data provided")
            return
        min_len = min(len(xs), len(ys))
        xs, ys = xs[:min_len], ys[:min_len]

        fig, ax = plt.subplots(figsize=(6, 4))
        if labels:
            unique_labels = list(dict.fromkeys(labels))
            for i, lab in enumerate(unique_labels):
                idx = [j for j, l in enumerate(labels[:min_len]) if l == lab]
                color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
                ax.scatter([xs[j] for j in idx], [ys[j] for j in idx],
                           color=color, s=40, label=lab, edgecolors="none")
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.scatter(xs, ys, color=_CHART_PALETTE[0], s=40, edgecolors="none")

        if trendline and len(xs) >= 2:
            import numpy as np
            coeffs = np.polyfit(xs, ys, 1)
            trend_x = [min(xs), max(xs)]
            trend_y = [coeffs[0] * x + coeffs[1] for x in trend_x]
            ax.plot(trend_x, trend_y, color=_MUTED, linestyle="--", linewidth=1.5)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", color=_HEADING, pad=10)

        run_dir = _run_dir_from_story(story)
        assets_dir = _ensure_assets_dir(run_dir)
        fname = _asset_hash("scatter", (title, xs, ys, labels, trendline))
        path = str(assets_dir / fname)
        if not os.path.isfile(path):
            _render_chart_to_png(fig, path)
        _embed_asset(story, path, width=width, caption=caption, default_width=_CHART_DEFAULT_W)
    except Exception as e:
        _warn(f"add_scatter failed: {e}")
        from notes_kit import add_paragraph
        add_paragraph(story, f"[chart render failed: {e}]")


def add_histogram(story: list, title: str | None = None,
                  data: list[float] | None = None, bins: int | list = 20,
                  xlabel: str | None = None, ylabel: str | None = None,
                  caption: str | None = None, width: float | None = None):
    """Append a histogram."""
    try:
        plt = _setup_mpl_theme()
        data = data or []
        if not data:
            _warn("add_histogram: no data provided")
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins=bins, color=_CHART_PALETTE[0], edgecolor="none", alpha=0.85)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", color=_HEADING, pad=10)

        run_dir = _run_dir_from_story(story)
        assets_dir = _ensure_assets_dir(run_dir)
        fname = _asset_hash("histogram", (title, len(data), str(bins)))
        path = str(assets_dir / fname)
        if not os.path.isfile(path):
            _render_chart_to_png(fig, path)
        _embed_asset(story, path, width=width, caption=caption, default_width=_CHART_DEFAULT_W)
    except Exception as e:
        _warn(f"add_histogram failed: {e}")
        from notes_kit import add_paragraph
        add_paragraph(story, f"[chart render failed: {e}]")


def add_box_plot(story: list, title: str | None = None,
                 groups: list[str] | None = None,
                 values: list[list[float]] | None = None,
                 xlabel: str | None = None, ylabel: str | None = None,
                 caption: str | None = None, width: float | None = None):
    """Append a grouped box plot."""
    try:
        plt = _setup_mpl_theme()
        groups = groups or []
        values = values or []
        if not values:
            _warn("add_box_plot: no data provided")
            return

        fig, ax = plt.subplots(figsize=(max(4, len(values) * 1.2), 4))
        bp = ax.boxplot(values, patch_artist=True)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(_CHART_PALETTE[i % len(_CHART_PALETTE)])
            box.set_alpha(0.6)
            box.set_edgecolor(_BORDER)
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color(_BORDER)
        for flier in bp["fliers"]:
            flier.set_markeredgecolor(_MUTED)

        if groups:
            ax.set_xticklabels(groups)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", color=_HEADING, pad=10)

        run_dir = _run_dir_from_story(story)
        assets_dir = _ensure_assets_dir(run_dir)
        fname = _asset_hash("box_plot", (title, groups, str(values)))
        path = str(assets_dir / fname)
        if not os.path.isfile(path):
            _render_chart_to_png(fig, path)
        _embed_asset(story, path, width=width, caption=caption, default_width=_CHART_DEFAULT_W)
    except Exception as e:
        _warn(f"add_box_plot failed: {e}")
        from notes_kit import add_paragraph
        add_paragraph(story, f"[chart render failed: {e}]")


def add_area_chart(story: list, title: str | None = None,
                   series: list[dict] | None = None,
                   xlabel: str | None = None, ylabel: str | None = None,
                   stacked: bool = True,
                   caption: str | None = None, width: float | None = None):
    """Append a stacked area chart."""
    try:
        plt = _setup_mpl_theme()
        series = series or []
        if not series:
            _warn("add_area_chart: no series provided")
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        for i, s in enumerate(series):
            color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
            xs = s.get("xs", list(range(len(s.get("ys", [])))))
            ys = s.get("ys", [])
            ax.fill_between(xs, ys, alpha=0.5, color=color, label=s.get("label", ""))
            ax.plot(xs, ys, color=color, linewidth=1.5)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", color=_HEADING, pad=10)
        if any(s.get("label") for s in series):
            ax.legend(frameon=False, fontsize=9)

        run_dir = _run_dir_from_story(story)
        assets_dir = _ensure_assets_dir(run_dir)
        fname = _asset_hash("area_chart", (title, str(series), stacked))
        path = str(assets_dir / fname)
        if not os.path.isfile(path):
            _render_chart_to_png(fig, path)
        _embed_asset(story, path, width=width, caption=caption, default_width=_CHART_DEFAULT_W)
    except Exception as e:
        _warn(f"add_area_chart failed: {e}")
        from notes_kit import add_paragraph
        add_paragraph(story, f"[chart render failed: {e}]")


def add_radar_chart(story: list, title: str | None = None,
                    categories: list[str] | None = None,
                    series: list[dict] | None = None,
                    values: list[float] | None = None,
                    fill: bool = True, alpha: float = 0.25,
                    caption: str | None = None, width: float | None = None):
    """Append a radar (spider) chart comparing multiple variables across axes.

    Single-series via `values=`, multi-series via `series=`.
    Each series dict: {"label": str, "values": list[float]}.
    """
    try:
        plt = _setup_mpl_theme()
        import numpy as np

        categories = categories or []
        if series is None and values is not None:
            series = [{"label": "", "values": values}]
        if not series:
            _warn("add_radar_chart: no data provided")
            return

        # Coerce lengths
        max_len = max(len(s.get("values", [])) for s in series)
        if not categories:
            categories = [f"V{i+1}" for i in range(max_len)]
        min_len = min(len(categories), max_len)
        categories = categories[:min_len]
        for s in series:
            vals = s.get("values", [])
            s["values"] = (vals + [0] * max_len)[:min_len]

        N = min_len
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(30)

        for i, s in enumerate(series):
            color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
            vals = s["values"][:min_len]
            vals_closed = vals + vals[:1]
            ax.plot(angles, vals_closed, color=color, linewidth=2,
                    label=s.get("label", ""))
            if fill:
                ax.fill(angles, vals_closed, color=color, alpha=alpha)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold",
                         color=_HEADING, y=1.08)
        if any(s.get("label") for s in series):
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                      frameon=False, fontsize=9)

        run_dir = _run_dir_from_story(story)
        assets_dir = _ensure_assets_dir(run_dir)
        fname = _asset_hash("radar_chart",
                            (title, categories, str(series), fill))
        path = str(assets_dir / fname)
        if not os.path.isfile(path):
            _render_chart_to_png(fig, path)
        _embed_asset(story, path, width=width, caption=caption,
                     default_width=_CHART_DEFAULT_W)
    except Exception as e:
        _warn(f"add_radar_chart failed: {e}")
        from notes_kit import add_paragraph
        add_paragraph(story, f"[chart render failed: {e}]")


def add_heatmap(story: list, title: str | None = None,
                matrix: list[list[float]] | None = None,
                row_labels: list[str] | None = None,
                col_labels: list[str] | None = None,
                annotate: bool = True,
                caption: str | None = None, width: float | None = None):
    """Append a heatmap with optional cell annotations."""
    try:
        plt = _setup_mpl_theme()
        import numpy as np
        matrix = matrix or []
        if not matrix:
            _warn("add_heatmap: no matrix provided")
            return

        data = np.array(matrix)
        fig, ax = plt.subplots(figsize=(max(4, data.shape[1] * 0.8),
                                        max(3, data.shape[0] * 0.6)))
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)

        if col_labels:
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, fontsize=8)
        if row_labels:
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=8)

        if annotate:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    color = "white" if val > data.max() * 0.6 else _INK
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=color)

        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", color=_HEADING, pad=10)

        run_dir = _run_dir_from_story(story)
        assets_dir = _ensure_assets_dir(run_dir)
        fname = _asset_hash("heatmap", (title, str(matrix), annotate))
        path = str(assets_dir / fname)
        if not os.path.isfile(path):
            _render_chart_to_png(fig, path)
        _embed_asset(story, path, width=width, caption=caption, default_width=_CHART_DEFAULT_W)
    except Exception as e:
        _warn(f"add_heatmap failed: {e}")
        from notes_kit import add_paragraph
        add_paragraph(story, f"[chart render failed: {e}]")


# ===========================================================================
# DIAGRAM HELPERS (graphviz)
# ===========================================================================


def _render_dot_to_png(dot_src: str, path: str):
    """Render DOT source to PNG via the graphviz Python package."""
    gv = _import_graphviz()
    src = gv.Source(dot_src)
    src.format = "png"
    # graphviz renders to `<filename>.png` automatically
    base = path.removesuffix(".png")
    src.render(base, cleanup=True)
    # graphviz appends .png to the base name
    rendered = base + ".png"
    if rendered != path and os.path.isfile(rendered):
        os.rename(rendered, path)


def _dot_placeholder(story: list, kind: str, caption: str | None = None,
                     width: float | None = None):
    """Insert a styled placeholder when graphviz dot is unavailable."""
    _warn(f"graphviz 'dot' binary not found; {kind} used a placeholder.")
    from notes_kit import add_paragraph
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors as c
    add_paragraph(story, f"[{kind}: requires system graphviz — install with: sudo apt install graphviz]")


def _dot_empty(story: list, kind: str, caption: str | None = None,
               width: float | None = None):
    """Insert a placeholder when no edges/data provided."""
    _warn(f"add_{kind}: no edges/data provided.")
    from notes_kit import add_paragraph
    add_paragraph(story, f"[{kind}: no data provided]")


def _normalize_edges(edges: list) -> list[tuple[str, str, str]]:
    """Normalize edges to (src, dst, label) tuples."""
    result = []
    for e in edges:
        if len(e) == 2:
            result.append((str(e[0]), str(e[1]), ""))
        elif len(e) >= 3:
            result.append((str(e[0]), str(e[1]), str(e[2])))
    return result


def _build_dot_graph(edges: list[tuple[str, str, str]],
                     node_labels: dict[str, str] | None = None,
                     rankdir: str = "TB",
                     node_shape: str = "box",
                     node_style: str = "rounded,filled",
                     node_fill: str = "#EFF6FF",
                     node_edge: str = "#2563EB",
                     edge_color: str = "#64748B",
                     directed: bool = True,
                     graph_attrs: dict | None = None,
                     node_attrs: dict | None = None) -> str:
    """Build a DOT source string from edges and options."""
    node_labels = node_labels or {}
    arrow = "->" if directed else "--"
    digraph = "digraph" if directed else "graph"

    lines = [f'{digraph} G {{']
    lines.append(f'  rankdir={rankdir};')
    lines.append('  bgcolor="transparent";')
    lines.append('  dpi="150";')
    lines.append('  size="7.5,10";')
    lines.append('  ratio="compress";')
    lines.append(f'  node [shape={node_shape} style="{node_style}" fillcolor="{node_fill}" color="{node_edge}" fontname="Helvetica" fontsize=10];')
    lines.append(f'  edge [color="{edge_color}" fontname="Helvetica" fontsize=9];')

    if graph_attrs:
        for k, v in graph_attrs.items():
            lines.append(f'  {k}="{v}";')
    if node_attrs:
        for k, v in node_attrs.items():
            lines.append(f'  {k}="{v}";')

    # Emit edges
    for src, dst, label in edges:
        src_label = node_labels.get(src, src)
        dst_label = node_labels.get(dst, dst)
        # Escape quotes
        src_label = src_label.replace('"', '\\"')
        dst_label = dst_label.replace('"', '\\"')
        edge_label = label.replace('"', '\\"') if label else ""
        line = f'  "{src}" {arrow} "{dst}"'
        if edge_label:
            line += f' [label="{edge_label}"]'
        line += ";"
        lines.append(line)

    lines.append("}")
    return "\n".join(lines)


def _embed_dot_diagram(story: list, dot_src: str, kind: str,
                       caption: str | None = None, width: float | None = None):
    """Build, render, and embed a graphviz diagram. Graceful on missing dot."""
    if not _check_dot_binary():
        _dot_placeholder(story, kind, caption, width)
        return

    run_dir = _run_dir_from_story(story)
    assets_dir = _ensure_assets_dir(run_dir)
    fname = _asset_hash(kind, (dot_src,))
    path = str(assets_dir / fname)

    if not os.path.isfile(path):
        try:
            _render_dot_to_png(dot_src, path)
        except Exception as e:
            _warn(f"add_{kind} render failed: {e}")
            from notes_kit import add_paragraph
            add_paragraph(story, f"[{kind} render failed: {e}]")
            return

    _embed_asset(story, path, width=width, caption=caption, default_width=_CHART_DEFAULT_W)


def add_flowchart(story: list, edges: list,
                  node_labels: dict[str, str] | None = None,
                  rankdir: str = "TB",
                  caption: str | None = None, width: float | None = None):
    """Append a flowchart (rounded boxes, decision diamonds auto-detected)."""
    if not edges:
        _dot_empty(story, "flowchart", caption, width)
        return
    norm = _normalize_edges(edges)
    # Detect decision nodes (label ends with ?)
    decision_nodes = set()
    if node_labels:
        for nid, label in node_labels.items():
            if label.rstrip().endswith("?"):
                decision_nodes.add(nid)

    # Build custom DOT with diamond shapes for decisions
    lines = ['digraph G {']
    lines.append(f'  rankdir={rankdir};')
    lines.append('  bgcolor="transparent";')
    lines.append('  dpi="150";')
    lines.append('  size="7.5,10";')
    lines.append('  ratio="compress";')
    lines.append('  node [style="rounded,filled" fillcolor="#EFF6FF" color="#2563EB" fontname="Helvetica" fontsize=10];')
    lines.append('  edge [color="#64748B" fontname="Helvetica" fontsize=9];')
    for src, dst, label in norm:
        sl = (node_labels or {}).get(src, src).replace('"', '\\"')
        dl = (node_labels or {}).get(dst, dst).replace('"', '\\"')
        # Emit node definitions
        src_shape = "diamond" if src in decision_nodes else "box"
        dst_shape = "diamond" if dst in decision_nodes else "box"
        lines.append(f'  "{src}" [shape={src_shape} label="{sl}"];')
        lines.append(f'  "{dst}" [shape={dst_shape} label="{dl}"];')
        edge = f'  "{src}" -> "{dst}"'
        if label:
            edge += f' [label="{label.replace(chr(34), chr(92)+chr(34))}"]'
        edge += ";"
        lines.append(edge)
    lines.append("}")

    _embed_dot_diagram(story, "\n".join(lines), "flowchart", caption, width)


def add_tree(story: list, edges: list,
             node_labels: dict[str, str] | None = None,
             caption: str | None = None, width: float | None = None):
    """Append a tree diagram (strict TB layout)."""
    if not edges:
        _dot_empty(story, "tree", caption, width)
        return
    norm = _normalize_edges(edges)
    dot = _build_dot_graph(norm, node_labels=node_labels, rankdir="TB",
                           node_shape="box", node_style="rounded,filled",
                           node_fill="#EFF6FF", node_edge="#2563EB",
                           directed=True)
    _embed_dot_diagram(story, dot, "tree", caption, width)


def add_state_diagram(story: list, edges: list,
                      node_labels: dict[str, str] | None = None,
                      initial: str | list[str] | None = None,
                      accepting: list[str] | None = None,
                      caption: str | None = None, width: float | None = None):
    """Append a state machine diagram."""
    if not edges:
        _dot_empty(story, "state_diagram", caption, width)
        return
    norm = _normalize_edges(edges)
    accepting = accepting or []
    if isinstance(initial, str):
        initial = [initial]

    lines = ['digraph G {']
    lines.append('  rankdir=LR;')
    lines.append('  bgcolor="transparent";')
    lines.append('  node [shape=circle style="filled" fillcolor="#EFF6FF" color="#2563EB" fontname="Helvetica" fontsize=10];')
    lines.append('  edge [color="#64748B" fontname="Helvetica" fontsize=9];')

    # Initial arrow
    if initial:
        for init in initial:
            lines.append('  __start [shape=point width=0];')
            lines.append(f'  __start -> "{init}";')

    # Accepting states
    for acc in accepting:
        lines.append(f'  "{acc}" [shape=doublecircle];')

    for src, dst, label in norm:
        sl = (node_labels or {}).get(src, src).replace('"', '\\"')
        dl = (node_labels or {}).get(dst, dst).replace('"', '\\"')
        lines.append(f'  "{src}" [label="{sl}"];')
        lines.append(f'  "{dst}" [label="{dl}"];')
        edge = f'  "{src}" -> "{dst}"'
        if label:
            edge += f' [label="{label.replace(chr(34), chr(92)+chr(34))}"]'
        edge += ";"
        lines.append(edge)
    lines.append("}")

    _embed_dot_diagram(story, "\n".join(lines), "state_diagram", caption, width)


def add_block_diagram(story: list, edges: list,
                      node_labels: dict[str, str] | None = None,
                      rankdir: str = "LR",
                      caption: str | None = None, width: float | None = None):
    """Append an architecture block diagram (rectangles, left-to-right)."""
    if not edges:
        _dot_empty(story, "block_diagram", caption, width)
        return
    norm = _normalize_edges(edges)
    dot = _build_dot_graph(norm, node_labels=node_labels, rankdir=rankdir,
                           node_shape="box", node_style="filled",
                           node_fill="#EFF6FF", node_edge="#2563EB",
                           directed=True)
    _embed_dot_diagram(story, dot, "block_diagram", caption, width)


def add_graph(story: list, edges: list,
              node_labels: dict[str, str] | None = None,
              layout: str = "neato",
              caption: str | None = None, width: float | None = None):
    """Append an undirected graph visualization."""
    if not edges:
        _dot_empty(story, "graph", caption, width)
        return
    norm = _normalize_edges(edges)
    dot = _build_dot_graph(norm, node_labels=node_labels, rankdir="TB",
                           node_shape="circle", node_style="filled",
                           node_fill="#EFF6FF", node_edge="#2563EB",
                           directed=False)
    # Use layout engine hint (neato, circo, fdp) via graphviz.Source.engine
    # We embed the layout name in a comment for the runner; graphviz picks it up via render(engine=)
    # For simplicity, we just use the default layout; the agent can pick layout via the DOT comment
    _embed_dot_diagram(story, dot, "graph", caption, width)


def add_call_graph(story: list, edges: list,
                   node_labels: dict[str, str] | None = None,
                   caption: str | None = None, width: float | None = None):
    """Append a directed call graph (function call relationships)."""
    if not edges:
        _dot_empty(story, "call_graph", caption, width)
        return
    norm = _normalize_edges(edges)
    dot = _build_dot_graph(norm, node_labels=node_labels, rankdir="TB",
                           node_shape="box", node_style="filled",
                           node_fill="#EFF6FF", node_edge="#2563EB",
                           directed=True)
    _embed_dot_diagram(story, dot, "call_graph", caption, width)


def add_venn(story: list, sets: dict[str, set] | None = None,
             labels: dict[str, str] | None = None,
             caption: str | None = None, width: float | None = None):
    """Append a 2- or 3-circle Venn diagram (matplotlib, not graphviz)."""
    try:
        plt = _setup_mpl_theme()
        sets = sets or {}
        if len(sets) < 2 or len(sets) > 3:
            _warn("add_venn: supports 2 or 3 sets only")
            from notes_kit import add_paragraph
            add_paragraph(story, "[venn: requires 2 or 3 sets]")
            return

        fig, ax = plt.subplots(figsize=(5, 5))
        set_keys = list(sets.keys())
        set_colors = [_CHART_PALETTE[0], _CHART_PALETTE[1], _CHART_PALETTE[2]]
        circle_kwargs = [
            {"center": (-0.5, 0), "radius": 1.2},
            {"center": (0.5, 0), "radius": 1.2},
        ]
        if len(sets) == 3:
            circle_kwargs.append({"center": (0, -0.8), "radius": 1.2})

        from matplotlib.patches import Circle
        circles = []
        for i, kw in enumerate(circle_kwargs):
            c = Circle(**kw, fill=True, alpha=0.35,
                       facecolor=set_colors[i], edgecolor=set_colors[i], linewidth=2)
            ax.add_patch(c)
            circles.append(c)

        # Label regions
        all_sets = [sets[k] for k in set_keys]
        if len(all_sets) == 2:
            a, b = all_sets
            regions = [
                (a - b, (-1.2, 0.2)),
                (b - a, (1.2, 0.2)),
                (a & b, (0, 0.2)),
            ]
        else:
            a, b, c = all_sets
            regions = [
                (a - b - c, (-1.2, 0.8)),
                (b - a - c, (1.2, 0.8)),
                (c - a - b, (0, -1.5)),
                ((a & b) - c, (0, 0.8)),
                ((a & c) - b, (-0.8, -0.4)),
                ((b & c) - a, (0.8, -0.4)),
                (a & b & c, (0, -0.2)),
            ]

        for region_set, pos in regions:
            count = len(region_set)
            if count > 0:
                ax.text(pos[0], pos[1], str(count), ha="center", va="center",
                        fontsize=14, fontweight="bold", color=_INK)

        # Legend
        if labels:
            for i, key in enumerate(set_keys):
                ax.text(-2.5, 1.5 - i * 0.4, f"{labels.get(key, key)}: {len(sets[key])}",
                        fontsize=9, color=set_colors[i], fontweight="bold")

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")
        if labels is None:
            for i, key in enumerate(set_keys):
                ax.text(circle_kwargs[i]["center"][0],
                        circle_kwargs[i]["center"][1] + 1.5,
                        key, ha="center", fontsize=10, color=set_colors[i], fontweight="bold")

        run_dir = _run_dir_from_story(story)
        assets_dir = _ensure_assets_dir(run_dir)
        fname = _asset_hash("venn", (str(sets),))
        path = str(assets_dir / fname)
        if not os.path.isfile(path):
            _render_chart_to_png(fig, path)
        _embed_asset(story, path, width=width, caption=caption, default_width=_CHART_DEFAULT_W)
    except Exception as e:
        _warn(f"add_venn failed: {e}")
        from notes_kit import add_paragraph
        add_paragraph(story, f"[venn render failed: {e}]")


def add_timeline(story: list, events: list[dict] | None = None,
                 caption: str | None = None, width: float | None = None):
    """Append a horizontal timeline of events."""
    try:
        plt = _setup_mpl_theme()
        events = events or []
        if not events:
            _warn("add_timeline: no events provided")
            return

        years = [e.get("year", 0) for e in events]
        labels = [e.get("label", str(e.get("year", ""))) for e in events]
        y_min, y_max = min(years), max(years)
        pad = max(1, (y_max - y_min) * 0.1)

        fig, ax = plt.subplots(figsize=(max(6, len(events) * 1.5), 2.5))
        ax.plot([y_min - pad, y_max + pad], [0, 0], color=_BORDER, linewidth=2, zorder=1)

        for i, (year, label) in enumerate(zip(years, labels)):
            y_offset = 0.3 if i % 2 == 0 else -0.3
            ax.scatter(year, 0, s=60, color=_CHART_PALETTE[i % len(_CHART_PALETTE)],
                       zorder=2, edgecolors="white", linewidth=1)
            ax.text(year, y_offset, label, ha="center", va="bottom" if y_offset > 0 else "top",
                    fontsize=8, color=_INK, rotation=0)

        ax.set_xlim(y_min - pad, y_max + pad)
        ax.set_ylim(-0.8, 0.8)
        ax.axis("off")

        run_dir = _run_dir_from_story(story)
        assets_dir = _ensure_assets_dir(run_dir)
        fname = _asset_hash("timeline", (str(events),))
        path = str(assets_dir / fname)
        if not os.path.isfile(path):
            _render_chart_to_png(fig, path)
        _embed_asset(story, path, width=width, caption=caption, default_width=_CHART_DEFAULT_W)
    except Exception as e:
        _warn(f"add_timeline failed: {e}")
        from notes_kit import add_paragraph
        add_paragraph(story, f"[timeline render failed: {e}]")


# ===========================================================================
# MATH HELPER (matplotlib mathtext)
# ===========================================================================


def add_image_rasterize(story: list, path: str,
                        width: float | None = None,
                        caption: str | None = None,
                        page: int = 1,
                        dpi: int = 200,
                        bg: str | None = "white"):
    """Embed a *non-bitmap* source by rasterizing it to a cached PNG first.

    Bridges the gap that ``add_image`` alone cannot handle:

    * **SVG**  → rendered via cairosvg (preferred), falling back to
      ``rsvg-convert`` / Inkscape CLI / ImageMagick if cairosvg is missing.
    * **PDF**  → the given page (1-indexed) is rasterized via pdf2image
      (already a project dependency; requires poppler-utils).

    Bitmaps (PNG/JPG/GIF/BMP/WEBP) are passed straight through to
    ``add_image`` without conversion — use that helper directly if you
    know the source is already a raster.

    The rasterized PNG is written to ``assets/`` keyed by a hash of
    ``(path, mtime, page, dpi, bg)``, so re-running build only re-rasters
    when the source changes (``--no-cache`` wipes everything as usual).

    Args:
        path:    Source file path. SVG, PDF, or any PIL-readable bitmap.
        width:   Target drawn width in points (auto-fit if None).
        caption: Optional caption under the image.
        page:    1-indexed page number — only used for PDF sources.
        dpi:     Rasterization resolution. 200 is sharp on screen and
                 matches the chart default. 300+ for print.
        bg:      Background fill applied during rasterization. SVGs and
                 PDFs are typically transparent; setting ``bg="white"``
                 (the default) keeps them readable on light themes.
                 Pass ``None`` to preserve transparency, or any hex color.

    Degrades gracefully: on a missing source file, unsupported extension,
    or rasterizer failure, a red placeholder paragraph is appended
    (same contract as ``add_image``) and a warning is recorded.
    """
    if not path or not isinstance(path, str):
        _warn("add_image_rasterize: empty path")
        from notes_kit import add_paragraph
        add_paragraph(story, "[add_image_rasterize: empty path]")
        return

    # Resolve relative paths against run_dir if CWD-based lookup fails
    src_path = path
    if not os.path.isfile(src_path) and not os.path.isabs(src_path):
        try:
            run_dir = _current_run_dir
            if run_dir:
                candidate = os.path.join(run_dir, src_path)
                if os.path.isfile(candidate):
                    src_path = candidate
        except Exception:
            pass

    if not os.path.isfile(src_path):
        _warn(f"add_image_rasterize: file not found: {path}")
        from notes_kit import add_image
        add_image(story, path, width=width, caption=caption)  # renders missing-image placeholder
        return

    ext = os.path.splitext(src_path)[1].lower()

    ext = os.path.splitext(path)[1].lower()

    # --- Bitmaps: pass straight through -------------------------------------
    if ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"):
        from notes_kit import add_image
        add_image(story, path, width=width, caption=caption)
        return

    # --- Vector/PDF: rasterize & cache --------------------------------------
    try:
        run_dir = _run_dir_from_story(story)
        assets_dir = _ensure_assets_dir(run_dir)
        src_mtime = os.path.getmtime(src_path)
        cache_key = (str(Path(src_path).resolve()), int(src_mtime), ext,
                     int(page), int(dpi), bg or "")
        fname = _asset_hash("rasterize", cache_key)
        out_path = str(assets_dir / fname)

        if not os.path.isfile(out_path):
            if ext == ".svg":
                _rasterize_svg(src_path, out_path, dpi=dpi, bg=bg)
            elif ext == ".pdf":
                _rasterize_pdf_page(src_path, out_path, page=page, dpi=dpi, bg=bg)
            else:
                _warn(f"add_image_rasterize: unsupported extension '{ext}' "
                      f"(only .svg and .pdf are vector-rastered; bitmaps "
                      f"are passed through)")
                from notes_kit import add_image
                add_image(story, src_path, width=width, caption=caption)
                return

        _embed_asset(story, out_path, width=width, caption=caption,
                     default_width=_CHART_DEFAULT_W)
    except Exception as e:
        _warn(f"add_image_rasterize failed for {path}: {type(e).__name__}: {e}")
        from notes_kit import add_paragraph
        add_paragraph(story, f"[rasterize failed: {path}] — {e}")


def _auto_crop_to_content(img, bg: str | None = "white",
                          threshold: int = 245, pad_frac: float = 0.02):
    """Crop an image to the bounding box of its non-background content.

    Used by rasterization helpers to trim empty page margins from PDF page
    renders and SVG exports so the embedded image shows only the actual
    content — figures, text, diagrams — rather than the full A4 page frame.

    Args:
        img:        PIL Image (RGB/RGBA/LA).
        bg:         Background color as hex string ("#FFFFFF" or "white").
                    If None, a luminance-based threshold is used instead.
        threshold:  Pixel luminance (0-255) below which a pixel counts as
                    "content". Pixels >= threshold are treated as background.
                    Default 245 (very light gray = background).
        pad_frac:   Fractional padding around the detected bounding box.
                    0.02 = 2% of image width/height on each side.

    Returns:
        PIL Image cropped to the content area, or the original image
        unchanged if no content was found (all-white/blank).
    """
    from PIL import Image as _PILImage
    import numpy as np

    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    arr = np.array(img)

    # Build a boolean mask: True where pixel is NOT background
    if img.mode == "L":
        # Grayscale — direct luminance comparison
        content_mask = arr < threshold
    else:
        # RGB/RGBA — luminance = 0.299R + 0.587G + 0.114B
        lum = (0.299 * arr[:, :, 0] +
               0.587 * arr[:, :, 1] +
               0.114 * arr[:, :, 2]).astype(np.uint8)
        content_mask = lum < threshold

    # Find bounding box of content
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    if not rows.any() or not cols.any():
        # All background — return unchanged
        return img

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add padding (proportional to image size)
    h, w = arr.shape[:2]
    pad_y = int(h * pad_frac)
    pad_x = int(w * pad_frac)
    rmin = max(0, rmin - pad_y)
    rmax = min(h - 1, rmax + pad_y)
    cmin = max(0, cmin - pad_x)
    cmax = min(w - 1, cmax + pad_x)

    # Don't crop if the content fills >90% of the page (likely full-page content)
    content_h = rmax - rmin + 1
    content_w = cmax - cmin + 1
    if content_h > h * 0.9 and content_w > w * 0.9:
        return img

    return img.crop((cmin, rmin, cmax + 1, rmax + 1))


def _rasterize_svg(src: str, out: str, dpi: int = 200, bg: str | None = "white"):
    """Render an SVG to PNG, auto-cropped to content. Tries cairosvg first, then system binaries."""
    _do_rasterize_svg(src, out, dpi=dpi, bg=bg)
    # Auto-crop: trim empty margins around the rendered SVG content
    try:
        from PIL import Image as _PILImage
        img = _PILImage.open(out)
        cropped = _auto_crop_to_content(img, bg=bg)
        if cropped is not img:
            cropped.save(out, "PNG")
    except Exception:
        pass  # non-fatal: keep the uncropped version


def _do_rasterize_svg(src: str, out: str, dpi: int = 200, bg: str | None = "white"):
    """Render SVG to PNG (raw, no post-processing). Tries backends in order."""
    # 1) cairosvg (pure-ish Python, no system binary needed)
    try:
        import cairosvg  # type: ignore
        bg_arg = {"background_color": bg} if bg else {}
        cairosvg.svg2png(url=src, write_to=out,
                         output_width=0, output_height=0,
                         scale=dpi / 96.0, **bg_arg)
        # scale=dpi/96 converts CSS px @96dpi to higher-res pixels
        return
    except ImportError:
        pass
    except Exception:
        # cairosvg present but failed — fall through to system binaries
        pass

    # 2) rsvg-convert (librsvg CLI)
    if shutil.which("rsvg-convert"):
        cmd = ["rsvg-convert", "-d", str(dpi), "-p", str(dpi)]
        if bg:
            cmd += ["-b", bg]
        cmd += ["-o", out, src]
        _run_subprocess(cmd, "rsvg-convert")
        return

    # 3) Inkscape CLI
    if shutil.which("inkscape"):
        cmd = ["inkscape", src,
               f"--export-dpi={dpi}",
               "--export-type=png",
               f"--export-filename={out}"]
        if bg:
            # Inkscape flag changed across versions; best-effort
            cmd.append(f"--export-background={bg}")
        _run_subprocess(cmd, "inkscape")
        return

    # 4) ImageMagick convert (last resort)
    if shutil.which("convert"):
        cmd = ["convert", "-density", str(dpi)]
        if bg:
            cmd += ["-background", bg, "-flatten"]
        cmd += [src, out]
        _run_subprocess(cmd, "convert")
        return

    raise RuntimeError(
        "no SVG rasterizer available (install cairosvg (`uv add cairosvg`) "
        "or system rsvg-convert / inkscape / imagemagick)")


def _rasterize_pdf_page(src: str, out: str, page: int = 1,
                        dpi: int = 200, bg: str | None = "white"):
    """Rasterize a single PDF page to PNG via pdf2image (needs poppler-utils).

    The rendered image is auto-cropped to the content bounding box so the
    embedded figure shows only the actual content area, not the full page
    margins, headers, and footers.
    """
    from pdf2image import convert_from_path
    from PIL import Image as _PILImage
    pages = convert_from_path(src, dpi=dpi, first_page=page,
                              last_page=page, fmt="png")
    if not pages:
        raise RuntimeError(f"pdf2image returned no pages for {src}")
    img = pages[0]
    if bg:
        if img.mode in ("RGBA", "LA", "P"):
            bg_rgb = _hex_to_rgb_tuple(bg) if bg.startswith("#") else (255, 255, 255)
            new = _PILImage.new("RGB", img.size, bg_rgb)
            new.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = new
    img = _auto_crop_to_content(img, bg=bg)
    img.save(out, "PNG")


def _hex_to_rgb_tuple(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(c * 2 for c in color)
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))


def _run_subprocess(cmd: list[str], tool_name: str):
    """Run a rasterization subprocess. Raises on non-zero exit."""
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{tool_name} failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )


def add_math(story: list, expr: str, display: bool = True,
             width: float | None = None, caption: str | None = None,
             fontsize: float | None = None):
    """Append a rendered LaTeX formula as a display equation.

    Args:
        expr:     LaTeX expression (without $ delimiters).
        display:  If True, render as display equation (larger, centered).
                  If False, render inline-sized.
        width:    Explicit drawn width in points. Auto-fit if None.
        caption:  Optional caption below the equation.
        fontsize: Override the rendered font size in points.
                  Defaults: 12pt for display, 10.5pt for inline.
                  Body text is 10.5pt — 12pt gives a ~1.14x ratio
                  matching standard scientific publishing conventions
                  (LaTeX \\displaystyle ≈ 1.2x body).
    """
    if not expr or not expr.strip():
        _warn("add_math: empty expression")
        return
    try:
        plt = _setup_mpl_theme()
        if fontsize is None:
            fontsize = 12 if display else 10.5
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${expr}$", fontsize=fontsize, ha="left", va="bottom")

        run_dir = _run_dir_from_story(story)
        assets_dir = _ensure_assets_dir(run_dir)
        fname = _asset_hash("math", (expr, display, fontsize))
        path = str(assets_dir / fname)
        if not os.path.isfile(path):
            # pad_inches=0 — no whitespace padding around the formula
            # (tighter layout; avoids the huge-gap-before-caption regression)
            fig.savefig(path, dpi=_MATH_DPI, bbox_inches="tight",
                        transparent=True, pad_inches=0)
        plt.close(fig)
        # Cap default width so long display formulas don't dominate the page:
        # display ≤ 60% of frame width, inline ≤ 40%.
        cap = _CHART_DEFAULT_W * (0.60 if display else 0.40)
        if width is not None and width > cap:
            width = cap
        _embed_asset(story, path, width=width, caption=caption,
                     default_width=cap)
    except Exception as e:
        _warn(f"add_math failed: {e}")
        # Fallback: show expression verbatim in monospace
        from notes_kit import add_paragraph
        add_paragraph(story, f"[math render failed: {e}] — expression: <font face='Courier'>{expr}</font>")


# ===========================================================================
# EXPORTS
# ===========================================================================

__all__ = [
    "add_bar_chart", "add_line_chart", "add_pie_chart", "add_donut_chart",
    "add_scatter", "add_histogram", "add_box_plot", "add_area_chart",
    "add_heatmap", "add_radar_chart",
    "add_flowchart", "add_tree", "add_state_diagram", "add_block_diagram",
    "add_graph", "add_call_graph", "add_venn", "add_timeline",
    "add_math",
    "add_image_rasterize",
    "get_graphics_warnings", "set_run_dir", "get_run_dir",
]
