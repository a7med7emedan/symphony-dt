from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from symphony_dt.logging import get_logger


@dataclass(frozen=True)
class OmicsConfig:
    """
    Implements Methods section 'Omics pathway scores as prior covariates'.

    Supports two input styles:

    1) Raw-feature draws:
       - multiple rows per patient with draw time relative to baseline
       - columns: patient_id, draw_time_days (<=0 allowed), plus features
       - selects latest pre-baseline draw (max draw_time_days <= 0)
       - harmonizes to training feature list, imputes missing at training mean,
         standardizes by training mean/sd
       - applies pathway membership vectors w_p (0/1) to produce P<=10 pathway scores z_i

    2) Precomputed pathway scores:
       - single row per patient containing columns named by z_prefix (default 'z_')
       - passes through these z columns and standardizes them using training-only SD (per Methods)

    If pathway membership is not provided and z_ columns do not exist, returns z=0 with omics_missing=True.
    """
    draw_time_col: str = "t_draw_days"
    patient_id_col: str = "patient_id"
    mode: str = "auto"  # auto|raw|z|none
    pathway_membership_path: Optional[str] = None  # CSV: pathway,feature,is_member
    max_pathways: int = 10
    min_genes_per_pathway: int = 5
    z_prefix: str = "z_"


@dataclass(frozen=True)
class OmicsArtifacts:
    z: pd.DataFrame
    omics_missing: pd.DataFrame
    training_feature_means: Dict[str, float]
    training_feature_sds: Dict[str, float]
    training_pathway_sds: Dict[str, float]
    pathway_features: Dict[str, List[str]]


def _detect_z_columns(df: pd.DataFrame, z_prefix: str) -> List[str]:
    return [c for c in df.columns if str(c).startswith(z_prefix)]


def _load_membership(path: Path) -> Dict[str, List[str]]:
    m = pd.read_csv(path)
    required = {"pathway", "feature", "is_member"}
    missing = sorted([c for c in required if c not in m.columns])
    if missing:
        raise ValueError(f"pathway membership file missing columns: {missing}")

    def to_bool(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce").fillna(0) != 0
        ss = s.astype(str).str.strip().str.lower()
        return ss.isin({"1", "true", "t", "yes", "y"})

    m = m.copy()
    m["is_member"] = to_bool(m["is_member"])
    m = m[m["is_member"]].copy()

    groups: Dict[str, List[str]] = {}
    for pth, sub in m.groupby("pathway"):
        groups[str(pth)] = sorted(sub["feature"].astype(str).unique().tolist())
    return groups


def _select_latest_prebaseline_draw(df: pd.DataFrame, cfg: OmicsConfig, patient_ids: Sequence[int]) -> pd.DataFrame:
    required = {cfg.patient_id_col, cfg.draw_time_col}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Omics raw mode requires columns: {missing}")

    d = df[df[cfg.patient_id_col].isin(list(patient_ids))].copy()
    d[cfg.patient_id_col] = pd.to_numeric(d[cfg.patient_id_col], errors="coerce").astype("Int64")
    d = d.dropna(subset=[cfg.patient_id_col]).copy()
    d[cfg.patient_id_col] = d[cfg.patient_id_col].astype(int)

    d[cfg.draw_time_col] = pd.to_numeric(d[cfg.draw_time_col], errors="coerce")
    d = d.dropna(subset=[cfg.draw_time_col]).copy()

    d = d[d[cfg.draw_time_col] <= 0].copy()
    if d.empty:
        return d

    d = d.sort_values([cfg.patient_id_col, cfg.draw_time_col], ascending=[True, True])
    idx = d.groupby(cfg.patient_id_col)[cfg.draw_time_col].idxmax()
    return d.loc[idx].sort_values(cfg.patient_id_col).reset_index(drop=True)


def _training_stats(X: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    means: Dict[str, float] = {}
    sds: Dict[str, float] = {}
    for c in X.columns:
        mu = float(pd.to_numeric(X[c], errors="coerce").mean())
        sd = float(pd.to_numeric(X[c], errors="coerce").std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        means[c] = mu
        sds[c] = sd
    return means, sds


def _standardize_features(X: pd.DataFrame, means: Dict[str, float], sds: Dict[str, float]) -> pd.DataFrame:
    Z = X.copy()
    for c in Z.columns:
        Z[c] = (pd.to_numeric(Z[c], errors="coerce") - means[c]) / sds[c]
    return Z


def _compute_pathway_means(
    X_std: pd.DataFrame,
    pathway_features: Dict[str, List[str]],
    min_genes: int,
    max_pathways: int,
) -> pd.DataFrame:
    eligible: List[Tuple[str, int]] = []
    for p, feats in pathway_features.items():
        feats2 = [f for f in feats if f in X_std.columns]
        if len(feats2) >= min_genes:
            eligible.append((p, len(feats2)))
    eligible.sort(key=lambda t: (-t[1], t[0]))
    selected_paths = [p for p, _ in eligible[:max_pathways]]

    out = pd.DataFrame(index=X_std.index)
    for p in selected_paths:
        feats2 = [f for f in pathway_features[p] if f in X_std.columns]
        out[p] = X_std[feats2].mean(axis=1)
    return out


def build_omics_covariates(
    omics: pd.DataFrame,
    patient_ids_all: Sequence[int],
    train_patient_ids: Sequence[int],
    cfg: OmicsConfig,
    output_dir: Optional[Path] = None,
) -> OmicsArtifacts:
    """
    Returns per-patient pathway scores z_i (P<=10) and missing indicator.
    Training-only computations for standardization are respected per Methods.
    """

    ids_all = sorted(set(int(x) for x in patient_ids_all))
    ids_train = sorted(set(int(x) for x in train_patient_ids))

    if omics is None or omics.empty:
        z = pd.DataFrame({"patient_id": ids_all})
        missing = pd.DataFrame({"patient_id": ids_all, "omics_missing": True})
        return OmicsArtifacts(z=z, omics_missing=missing, training_feature_means={}, training_feature_sds={}, training_pathway_sds={}, pathway_features={})

    df = omics.copy()

    z_cols = _detect_z_columns(df, cfg.z_prefix)
    mode = cfg.mode
    if mode == "auto":
        if z_cols:
            mode = "z"
        elif cfg.pathway_membership_path is not None:
            mode = "raw"
        else:
            mode = "none"

    if mode == "z":
        if cfg.patient_id_col not in df.columns:
            raise ValueError("Omics z mode requires patient_id column")
        df[cfg.patient_id_col] = pd.to_numeric(df[cfg.patient_id_col], errors="coerce").astype("Int64")
        df = df.dropna(subset=[cfg.patient_id_col]).copy()
        df[cfg.patient_id_col] = df[cfg.patient_id_col].astype(int)
        df = df[df[cfg.patient_id_col].isin(ids_all)].copy()
        df = df.sort_values(cfg.patient_id_col).drop_duplicates(subset=[cfg.patient_id_col], keep="first")

        z_cols = _detect_z_columns(df, cfg.z_prefix)
        if not z_cols:
            mode = "none"
        else:
            Zraw = df.set_index(cfg.patient_id_col)[z_cols].apply(pd.to_numeric, errors="coerce")
            Ztrain = Zraw.loc[[pid for pid in ids_train if pid in Zraw.index]]
            pathway_sds: Dict[str, float] = {}
            for c in z_cols:
                sd = float(Ztrain[c].std(ddof=0))
                if not np.isfinite(sd) or sd <= 0:
                    sd = 1.0
                pathway_sds[c] = sd
            Zstd = Zraw.copy()
            for c in z_cols:
                Zstd[c] = Zstd[c] / pathway_sds[c]

            Zstd = Zstd.reindex(ids_all).fillna(0.0)
            out_z = Zstd.reset_index().rename(columns={cfg.patient_id_col: "patient_id"})
            if "patient_id" not in out_z.columns:
                out_z = out_z.rename(columns={out_z.columns[0]: "patient_id"})

            missing = pd.DataFrame({"patient_id": ids_all, "omics_missing": ~pd.Series(ids_all).isin(Zraw.index).to_numpy()})
            if output_dir is not None:
                import json
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "omics_training_stats.json").write_text(json.dumps({"mode": "z", "pathway_sds": pathway_sds}, indent=2, sort_keys=True), encoding="utf-8")
            return OmicsArtifacts(z=out_z, omics_missing=missing, training_feature_means={}, training_feature_sds={}, training_pathway_sds=pathway_sds, pathway_features={})

    if mode == "raw":
        if cfg.pathway_membership_path is None:
            raise ValueError("raw mode requires pathway_membership_path")
        membership = _load_membership(Path(cfg.pathway_membership_path))

        sel = _select_latest_prebaseline_draw(df, cfg, ids_all)
        feat_cols = [c for c in sel.columns if c not in {cfg.patient_id_col, cfg.draw_time_col}]
        if not feat_cols:
            mode = "none"
        else:
            X = sel.set_index(cfg.patient_id_col)[feat_cols].apply(pd.to_numeric, errors="coerce")
            X_train = X.loc[[pid for pid in ids_train if pid in X.index]]
            means, sds = _training_stats(X_train)

            full_cols = sorted(means.keys())
            X_full = X.reindex(columns=full_cols)
            for c in full_cols:
                X_full[c] = X_full[c].fillna(means[c])
            X_std = _standardize_features(X_full, means, sds)

            M = _compute_pathway_means(X_std, membership, cfg.min_genes_per_pathway, cfg.max_pathways)

            M_train = M.loc[[pid for pid in ids_train if pid in M.index]]
            pathway_sds: Dict[str, float] = {}
            for p in M.columns:
                sd = float(M_train[p].std(ddof=0))
                if not np.isfinite(sd) or sd <= 0:
                    sd = 1.0
                pathway_sds[p] = sd

            Z = M.copy()
            for p in M.columns:
                Z[p] = Z[p] / pathway_sds[p]
            Z = Z.rename(columns={p: f"{cfg.z_prefix}{p}" for p in Z.columns})

            Z = Z.reindex(ids_all).fillna(0.0)
            out_z = Z.reset_index().rename(columns={"index": "patient_id"})
            missing = pd.DataFrame({"patient_id": ids_all, "omics_missing": ~pd.Series(ids_all).isin(sel[cfg.patient_id_col].unique()).to_numpy()})

            if output_dir is not None:
                import json
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "omics_training_stats.json").write_text(json.dumps({"mode": "raw", "feature_means": means, "feature_sds": sds, "pathway_sds": pathway_sds}, indent=2, sort_keys=True), encoding="utf-8")
                (output_dir / "omics_pathways.json").write_text(json.dumps({k: [f for f in v if f in full_cols] for k, v in membership.items()}, indent=2, sort_keys=True), encoding="utf-8")

            return OmicsArtifacts(z=out_z, omics_missing=missing, training_feature_means=means, training_feature_sds=sds, training_pathway_sds=pathway_sds, pathway_features={k: [f for f in v if f in full_cols] for k, v in membership.items()})

    z = pd.DataFrame({"patient_id": ids_all})
    missing = pd.DataFrame({"patient_id": ids_all, "omics_missing": True})
    return OmicsArtifacts(z=z, omics_missing=missing, training_feature_means={}, training_feature_sds={}, training_pathway_sds={}, pathway_features={})
