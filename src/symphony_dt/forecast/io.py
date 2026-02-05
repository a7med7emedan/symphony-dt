from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def write_table_atomic(path: Path, df: pd.DataFrame) -> None:
    """Shard-safe atomic write: write to temp then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def aggregate_tables(inputs: List[Path], out_path: Path, dedup_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """Concatenate CSVs and optionally drop duplicates by dedup_keys."""
    dfs = [pd.read_csv(p) for p in inputs if p.exists()]
    if not dfs:
        raise FileNotFoundError("No input tables found to aggregate.")
    out = pd.concat(dfs, ignore_index=True)
    if dedup_keys:
        out = out.drop_duplicates(subset=dedup_keys)
    write_table_atomic(out_path, out)
    return out
