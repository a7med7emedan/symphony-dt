from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from symphony_dt.paths import ensure_dir


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    metadata: Dict[str, Any]


def _utc_timestamp_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_run_context(
    outputs_root: Path,
    project: str,
    seed: int,
    overwrite: bool,
    mode: str,
    shard_id: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> RunContext:
    ensure_dir(outputs_root)
    stamp = _utc_timestamp_compact()
    run_id = f"{project}__{stamp}__seed{seed}__{mode}"
    if shard_id is not None and num_shards is not None:
        run_id += f"__shard{shard_id}of{num_shards}"

    run_dir = outputs_root / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"Run directory exists and overwrite=false: {run_dir}")
    ensure_dir(run_dir)

    metadata: Dict[str, Any] = {
        "project": project,
        "run_id": run_id,
        "created_utc": stamp,
        "seed": seed,
        "mode": mode,
        "shard_id": shard_id,
        "num_shards": num_shards,
    }
    return RunContext(run_dir=run_dir, metadata=metadata)


def export_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def export_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def export_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
