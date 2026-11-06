from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class ColumnRule:
    name: str
    dtype: Any
    nullable: bool = True
    allowed: Optional[Sequence[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    # Deprecated names accepted in place of `name`, for tables written by an earlier release.
    aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TableContract:
    table_name: str
    required: Tuple[ColumnRule, ...]
    primary_key: Tuple[str, ...] = ()

    def required_names(self) -> List[str]:
        return [c.name for c in self.required]


def _coerce_series(s: pd.Series, dtype: Any) -> pd.Series:
    if dtype in (int, "int64", "Int64"):
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    if dtype in (float, "float64"):
        return pd.to_numeric(s, errors="coerce").astype(float)
    if dtype in (bool, "bool"):
        if s.dtype == bool:
            return s.astype(bool)
        if pd.api.types.is_numeric_dtype(s):
            return (pd.to_numeric(s, errors="coerce").fillna(0) != 0)
        ss = s.astype(str).str.strip().str.lower()
        return ss.isin({"true", "1", "t", "yes", "y"})
    if dtype in (str, "string"):
        return s.astype("string")
    return s.astype(dtype)


def _resolve_aliases(df: pd.DataFrame, contract: TableContract) -> pd.DataFrame:
    """Fill each required column from its first present alias when the canonical name is absent."""
    out = df
    for rule in contract.required:
        if rule.name in out.columns:
            continue
        for alias in rule.aliases:
            if alias in out.columns:
                if out is df:
                    out = df.copy()
                out[rule.name] = out[alias]
                break
    return out


def validate_table(df: pd.DataFrame, contract: TableContract) -> pd.DataFrame:
    df = _resolve_aliases(df, contract)
    missing = [c for c in contract.required_names() if c not in df.columns]
    if missing:
        raise ValueError(f"{contract.table_name}: missing required columns: {missing}")

    out = df.copy()
    errors: List[str] = []

    for rule in contract.required:
        try:
            out[rule.name] = _coerce_series(out[rule.name], rule.dtype)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{contract.table_name}.{rule.name}: failed to coerce to {rule.dtype}: {e}")

        if not rule.nullable and out[rule.name].isna().any():
            errors.append(f"{contract.table_name}.{rule.name}: contains nulls but nullable=false")

        if rule.allowed is not None:
            allowed_set = set(rule.allowed)
            vals = set(out[rule.name].dropna().unique().tolist())
            bad = sorted([v for v in vals if v not in allowed_set])
            if bad:
                errors.append(f"{contract.table_name}.{rule.name}: unexpected values {bad} (allowed={sorted(allowed_set)})")

        if rule.min_value is not None or rule.max_value is not None:
            x = pd.to_numeric(out[rule.name], errors="coerce")
            if rule.min_value is not None and (x.dropna() < rule.min_value).any():
                errors.append(f"{contract.table_name}.{rule.name}: has values < {rule.min_value}")
            if rule.max_value is not None and (x.dropna() > rule.max_value).any():
                errors.append(f"{contract.table_name}.{rule.name}: has values > {rule.max_value}")

    if contract.primary_key:
        dup = out.duplicated(subset=list(contract.primary_key), keep=False)
        if dup.any():
            errors.append(f"{contract.table_name}: primary_key not unique; duplicated rows={int(dup.sum())}")

    if errors:
        raise ValueError("Schema validation failed:\n- " + "\n- ".join(errors))
    return out
