"""Tidy output tables a modeller can take into R, Python, Excel or a manuscript pipeline.

Every table is long and tidy: one row per observation unit, one column per variable. `write_tables` writes a
dict of tables as CSV and Parquet files and one Excel workbook with a sheet per table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from symphony_dt import constants as C
from symphony_dt.labels import validation

__all__ = ["forecast_table", "posterior_summary", "scores_table", "tidy_summary", "write_tables"]

_SUMMARY_COLUMNS = {
    "Mean": "mean",
    "StdDev": "sd",
    "MCSE": "mcse",
    "5%": "q5",
    "50%": "q50",
    "95%": "q95",
    "ESS_bulk": "ess_bulk",
    "ESS_tail": "ess_tail",
    "R_hat": "rhat",
}


@validation("TESTED")
def posterior_summary(fit: Any, *, globals_only: bool = True, retained_only: bool = True) -> pd.DataFrame:
    """Return a tidy posterior summary from CmdStan's stansummary, on the retained chains by default.

    Columns: parameter, mean, sd, mcse, q5, q50, q95, ess_bulk, ess_tail, rhat.

    Validation: TESTED.
    """
    from symphony_dt.diagnostics import read_sampler_columns, retained_chains, summary_on_chains

    files = list(fit.csv_files)
    keep = retained_chains(read_sampler_columns(files)["lp__"])["keep_index"] if retained_only else None
    table = summary_on_chains(files, keep)
    return tidy_summary(table, globals_only=globals_only)


@validation("TESTED")
def tidy_summary(table: pd.DataFrame, *, globals_only: bool = True) -> pd.DataFrame:
    """Turn a stansummary table into the tidy layout of `posterior_summary`.

    Validation: TESTED.
    """
    from symphony_dt.diagnostics import is_global

    rows = table.loc[[i for i in table.index if (is_global(str(i)) or not globals_only) and not str(i).endswith("__")]]
    out = rows.rename(columns=_SUMMARY_COLUMNS)[
        [c for c in _SUMMARY_COLUMNS.values() if c in rows.rename(columns=_SUMMARY_COLUMNS).columns]
    ]
    return out.rename_axis("parameter").reset_index()


@validation("TESTED")
def forecast_table(
    forecast: Any,
    participant_ids: Sequence[str] | None = None,
    quantiles: Sequence[float] = (0.025, 0.25, 0.5, 0.75, 0.975),
) -> pd.DataFrame:
    """Return one row per forecast target with endpoint name, horizon, observed value and quantiles.

    `participant_ids` maps the zero-based index `i` of an exact-route forecast to identifiers.

    Validation: TESTED.
    """
    targets = forecast.targets.reset_index(drop=True).copy()
    q = np.quantile(forecast.draws, list(quantiles), axis=0)
    for value, row in zip(quantiles, q, strict=True):
        targets[f"q{value:g}".replace("0.", "")] = row
    targets["mean"] = forecast.draws.mean(axis=0)
    targets["sd"] = forecast.draws.std(axis=0, ddof=1)
    if "endpoint" not in targets.columns:
        targets["endpoint"] = [C.ENDPOINTS[int(k) - 1] for k in targets["k"]]
    if participant_ids is not None and "i" in targets.columns and "participant_id" not in targets.columns:
        targets["participant_id"] = [participant_ids[int(i)] for i in targets["i"]]
    front = [c for c in ("participant_id", "i", "endpoint", "k", "t", "horizon", "y") if c in targets.columns]
    result: pd.DataFrame = targets[front + [c for c in targets.columns if c not in front]]
    return result


@validation("TESTED")
def scores_table(scores: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten a scores dict (FoldScores.scores or PooledScores.scores) to rows of cell, method, metric.

    Each row holds the estimate and its bootstrap interval.

    Validation: TESTED.
    """
    rows = []
    for key, cell in scores.get("cells", {}).items():
        endpoint, window = key.split("|")
        for method in ("twin_all", "twin", "locf", "lmm"):
            for metric, triple in cell.get(method, {}).items():
                rows.append(
                    {
                        "endpoint": endpoint,
                        "window": window,
                        "method": method,
                        "metric": metric,
                        "estimate": triple[0],
                        "lower": triple[1],
                        "upper": triple[2],
                        "n": cell.get("n_common" if method != "twin_all" else "n_all"),
                    }
                )
        for diff in ("crps_diff_twin_minus_locf", "crps_diff_twin_minus_lmm"):
            if diff in cell:
                rows.append(
                    {
                        "endpoint": endpoint,
                        "window": window,
                        "method": "paired",
                        "metric": diff,
                        "estimate": cell[diff][0],
                        "lower": cell[diff][1],
                        "upper": cell[diff][2],
                        "n": cell.get("n_common"),
                    }
                )
    for key, ev in scores.get("events", {}).items():
        endpoint, window = key.split("|")
        for metric in ("brier", "auc"):
            for method, triple in ev.get(metric, {}).items():
                rows.append(
                    {
                        "endpoint": endpoint,
                        "window": window,
                        "method": method,
                        "metric": metric,
                        "estimate": triple[0],
                        "lower": triple[1],
                        "upper": triple[2],
                        "n": ev.get("n"),
                    }
                )
        for metric in ("bss_twin_vs_locf", "bss_twin_vs_lmm", "prevalence"):
            if metric in ev:
                rows.append(
                    {
                        "endpoint": endpoint,
                        "window": window,
                        "method": "events",
                        "metric": metric,
                        "estimate": ev[metric][0],
                        "lower": ev[metric][1],
                        "upper": ev[metric][2],
                        "n": ev.get("n"),
                    }
                )
    return pd.DataFrame(rows)


@validation("TESTED")
def write_tables(
    tables: Mapping[str, pd.DataFrame],
    directory: str | Path,
    formats: Sequence[str] = ("csv", "parquet", "xlsx"),
) -> dict[str, list[str]]:
    """Write each table in each format; `xlsx` puts every table in one workbook, one sheet each.

    Returns the written paths per table.

    Validation: TESTED.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {name: [] for name in tables}
    for name, frame in tables.items():
        if "csv" in formats:
            frame.to_csv(out / f"{name}.csv", index=False)
            written[name].append(str(out / f"{name}.csv"))
        if "parquet" in formats:
            frame.to_parquet(out / f"{name}.parquet", index=False)
            written[name].append(str(out / f"{name}.parquet"))
    if "xlsx" in formats and tables:
        book = out / "tables.xlsx"
        with pd.ExcelWriter(book) as writer:
            for name, frame in tables.items():
                frame.to_excel(writer, sheet_name=name[:31], index=False)
        for name in tables:
            written[name].append(str(book))
    return written
