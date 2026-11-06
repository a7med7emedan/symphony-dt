"""Readers and writers for tables and datasets: CSV, TSV, Parquet, Excel, Arrow and polars frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from symphony_dt.data.dataset import TwinDataset
from symphony_dt.data.instruments import Instrument, InstrumentRegistry
from symphony_dt.labels import validation

__all__ = ["read_dataset", "read_table", "to_pandas", "write_dataset", "write_table"]

_FORMATS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "tsv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".xlsx": "excel",
    ".xls": "excel",
    ".arrow": "arrow",
    ".feather": "arrow",
    ".ipc": "arrow",
}


def _format(path: Path, fmt: str | None) -> str:
    if fmt is not None:
        return fmt
    try:
        return _FORMATS[path.suffix.lower()]
    except KeyError:
        raise ValueError(f"cannot infer the format of {path.name}; pass format=") from None


@validation("TESTED")
def to_pandas(frame: Any) -> pd.DataFrame:
    """Return a pandas DataFrame from a pandas, polars or pyarrow table.

    Validation: TESTED.
    """
    if isinstance(frame, pd.DataFrame):
        return frame
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()  # type: ignore[no-any-return]
    raise TypeError(f"cannot convert {type(frame).__name__} to a pandas DataFrame")


@validation("TESTED")
def read_table(path: str | Path, format: str | None = None, **kwargs: Any) -> pd.DataFrame:
    """Read a table from CSV, TSV, Parquet, Excel or Arrow, by suffix unless `format` is given.

    Validation: TESTED.
    """
    p = Path(path)
    fmt = _format(p, format)
    if fmt == "csv":
        frame: pd.DataFrame = pd.read_csv(p, **kwargs)
        return frame
    if fmt == "tsv":
        frame = pd.read_csv(p, sep="\t", **kwargs)
        return frame
    if fmt == "parquet":
        frame = pd.read_parquet(p, **kwargs)
        return frame
    if fmt == "excel":
        frame = pd.read_excel(p, **kwargs)
        return frame
    if fmt == "arrow":
        frame = pd.read_feather(p, **kwargs)
        return frame
    raise ValueError(f"unknown format {fmt!r}")


@validation("TESTED")
def write_table(frame: Any, path: str | Path, format: str | None = None) -> Path:
    """Write a pandas or polars table to CSV, TSV, Parquet, Excel or Arrow.

    Validation: TESTED.
    """
    df = to_pandas(frame)
    p = Path(path)
    fmt = _format(p, format)
    if fmt == "csv":
        df.to_csv(p, index=False)
    elif fmt == "tsv":
        df.to_csv(p, sep="\t", index=False)
    elif fmt == "parquet":
        df.to_parquet(p, index=False)
    elif fmt == "excel":
        df.to_excel(p, index=False)
    elif fmt == "arrow":
        df.reset_index(drop=True).to_feather(p)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    return p


@validation("TESTED")
def write_dataset(dataset: TwinDataset, directory: str | Path, format: str = "parquet") -> Path:
    """Write a dataset as one table per frame plus `dataset.json` with instruments and metadata.

    Validation: TESTED.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    suffix = {"parquet": ".parquet", "csv": ".csv", "arrow": ".arrow", "tsv": ".tsv"}[format]
    write_table(dataset.participants, out / f"participants{suffix}")
    write_table(dataset.observations, out / f"observations{suffix}")
    if dataset.visits is not None:
        write_table(dataset.visits, out / f"visits{suffix}")
    write_table(dataset.markers, out / f"markers{suffix}")
    meta = {
        "format": format,
        "instruments": [i.model_dump() for i in dataset.instruments],
        "metadata": dataset.metadata,
        "fingerprint": dataset.fingerprint(),
    }
    (out / "dataset.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return out


@validation("TESTED")
def read_dataset(directory: str | Path) -> TwinDataset:
    """Read a dataset written by `write_dataset` and check its fingerprint.

    Validation: TESTED.
    """
    src = Path(directory)
    meta = json.loads((src / "dataset.json").read_text())
    suffix = {"parquet": ".parquet", "csv": ".csv", "arrow": ".arrow", "tsv": ".tsv"}[meta["format"]]

    def load(name: str) -> pd.DataFrame | None:
        p = src / f"{name}{suffix}"
        if not p.exists():
            return None
        frame = read_table(p)
        if "participant_id" in frame.columns:
            frame["participant_id"] = frame["participant_id"].astype(str)
        return frame

    participants, observations = load("participants"), load("observations")
    if participants is None or observations is None:
        raise FileNotFoundError(f"{src} lacks participants or observations")
    instruments = InstrumentRegistry(Instrument(**i) for i in meta["instruments"])
    ds = TwinDataset(
        participants, observations, load("visits"), load("markers"), instruments, meta.get("metadata")
    )
    if ds.fingerprint() != meta["fingerprint"]:
        raise ValueError(f"{src}: dataset fingerprint differs from dataset.json")
    return ds
