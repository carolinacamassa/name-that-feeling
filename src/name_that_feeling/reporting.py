"""Export exhibit charts from marimo notebooks to SVG plus a per-experiment manifest.

Convention (see CLAUDE.md): notebooks save *exhibits* — charts that state a result
and are fully determined by the experiment's ``data/`` — and never *instruments*
(anything downstream of a ``mo.ui`` value). Slide decks are built from the
``figures/manifest.json`` files this module writes, never from live notebooks.
"""

import json
from datetime import date
from pathlib import Path


def save_chart(chart, name: str, *, caption: str, takeaway: str, notebook: str, params: dict | None = None):
    """Save an exhibit to ``<notebook dir>/figures/<name>.svg`` and record it in the manifest.

    Returns the chart unchanged, so a cell can end with ``save_chart(...)`` and still
    display it. ``notebook`` is the calling notebook's ``__file__`` (locates the figures
    dir and is recorded as the figure's source). ``caption`` says what the figure shows;
    ``takeaway`` states the claim it supports — decks quote these verbatim, so pin exact
    numbers here. ``params`` records pinned parameter values for promoted ``example_*``
    figures (see the exhibits-not-instruments rule).
    """
    exportable = chart if hasattr(chart, "save") else chart._chart  # unwrap mo.ui.altair_chart
    figures_dir = Path(notebook).resolve().parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    exportable.save(str(figures_dir / f"{name}.svg"))  # altair renders SVG via vl-convert, no browser

    manifest_path = figures_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[name] = {
        "file": f"{name}.svg",
        "caption": caption,
        "takeaway": takeaway,
        "notebook": Path(notebook).name,
        **({"params": params} if params else {}),
        "updated": date.today().isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return chart
