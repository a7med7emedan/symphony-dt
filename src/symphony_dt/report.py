"""Self-contained HTML reports from manifests, gate reports, scores and figures."""

from __future__ import annotations

import base64
import html
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from symphony_dt._version import __version__
from symphony_dt.labels import validation

__all__ = ["figure_to_html", "html_report", "write_html_report"]

_STYLE = (
    "body{font-family:system-ui,-apple-system,sans-serif;max-width:64rem;margin:2rem auto;padding:0 1rem;"
    "color:#1b1b1b}h1{color:#0072B2}table{border-collapse:collapse;margin:0.5rem 0 1.5rem}"
    "th,td{border:1px solid #ddd;padding:0.25rem 0.5rem;text-align:left;vertical-align:top}"
    "code{font-size:0.85em;white-space:pre-wrap}.pass{color:#009E73}.fail{color:#D55E00}"
    "footer{margin-top:3rem;color:#666;font-size:0.85em}"
)


@validation("TESTED")
def figure_to_html(fig: Any, *, alt: str = "figure") -> str:
    """Return an <img> tag embedding a matplotlib figure as base64 PNG.

    Validation: TESTED.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    data = base64.b64encode(buffer.getvalue()).decode()
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}" style="max-width:100%">'


def _table(d: Mapping[str, Any]) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td><code>"
        f"{html.escape(v if isinstance(v, str) else json.dumps(v, default=str))}</code></td></tr>"
        for k, v in d.items()
    )
    return f"<table>{rows}</table>"


@validation("TESTED")
def html_report(
    manifests: Sequence[Mapping[str, Any]] = (),
    *,
    title: str = "symphony-dt report",
    gates: Mapping[str, Any] | None = None,
    scores: Mapping[str, Any] | None = None,
    figures: Sequence[tuple[str, Any]] = (),
    notes: Sequence[str] = (),
) -> str:
    """Return one HTML page with run manifests, a gate verdict, score cells and figures.

    Validation: TESTED.
    """
    parts = [f"<h1>{html.escape(title)}</h1>"]
    parts += [f"<p>{html.escape(n)}</p>" for n in notes]
    if gates is not None:
        verdict = bool(gates.get("converged"))
        parts.append(
            f"<h2>Convergence gates</h2><p class='{'pass' if verdict else 'fail'}'>CONVERGED {verdict}</p>"
        )
        parts.append(_table({k: v for k, v in gates.items() if k not in ("diagnostics", "chain_exclusion")}))
    if scores is not None:
        parts.append("<h2>Scores</h2>")
        rows = []
        for key, cell in scores.get("cells", {}).items():
            twin = cell.get("twin", cell.get("twin_all", {}))
            rows.append(
                f"<tr><td>{html.escape(key)}</td><td>{cell.get('n_common', cell.get('n_all'))}</td>"
                f"<td>{twin.get('c95')}</td><td>{twin.get('c50')}</td><td>{twin.get('crps')}</td>"
                f"<td>{cell.get('crps_diff_twin_minus_locf')}</td>"
                f"<td>{cell.get('crps_diff_twin_minus_lmm')}</td></tr>"
            )
        parts.append(
            "<table><tr><th>cell</th><th>n</th><th>twin 95%</th><th>twin 50%</th>"
            "<th>twin CRPS</th><th>CRPS twin minus LOCF</th><th>CRPS twin minus LMM</th></tr>"
            + "".join(rows)
            + "</table>"
        )
    for caption, fig in figures:
        parts.append(
            f"<figure>{figure_to_html(fig, alt=caption)}<figcaption>{html.escape(caption)}"
            "</figcaption></figure>"
        )
    for i, m in enumerate(manifests, start=1):
        parts.append(f"<h2>Run manifest {i}: {html.escape(str(m.get('kind', '')))}</h2>")
        parts.append(_table({k: v for k, v in m.items() if k != "environment"}))
        if "environment" in m:
            parts.append(_table({"environment": m["environment"]}))
    parts.append(
        f"<footer>symphony-dt {__version__}. Developed by Ahmed Hemedan. Values are associational; "
        "no causal, treatment or deployment claim is made.</footer>"
    )
    return (
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{_STYLE}</style></head><body>{''.join(parts)}</body></html>\n"
    )


@validation("TESTED")
def write_html_report(path: str | Path, manifests: Sequence[Mapping[str, Any]] = (), **kwargs: Any) -> Path:
    """Write `html_report(...)` to `path` and return the path.

    Validation: TESTED.
    """
    target = Path(path)
    target.write_text(html_report(manifests, **kwargs))
    return target
