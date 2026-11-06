"""Deterministic synthetic inputs shared by the golden fixture generator and the golden tests.

Every case is built from the package simulator with fixed seeds, so the inputs are regenerated
identically wherever the tests run. Only the outputs of the code of record are committed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from symphony_dt.data import Standardisation, simulate_cohort, synthetic_visit_patterns
from symphony_dt.data.simulate import draw_prior
from symphony_dt.forecast import split_at_anchor

PATHWAY = ["pathway_1", "pathway_2"]
FIXTURES = Path(__file__).resolve().parents[1] / "data" / "golden"


def stan_data_case(seed: int = 101, n: int = 40):
    """A cohort with pathway markers (some missing) and five folds."""
    cohort = simulate_cohort(n, seed=seed, n_pathway=2)
    ds = cohort.dataset
    rng = np.random.default_rng(seed + 1)
    markers = ds.markers.copy()
    drop = rng.uniform(size=len(markers)) < 0.15
    ds.markers = markers[~drop].reset_index(drop=True)
    ids = ds.participant_ids
    folds = pd.DataFrame(
        {
            "patno": [int(i) for i in ids],
            "fold": rng.integers(1, 6, size=len(ids)),
            "cohort": ds.participants.set_index("participant_id").loc[ids, "cohort"].to_numpy(),
            "sex": ds.participants.set_index("participant_id").loc[ids, "sex"].to_numpy(),
        }
    )
    train = [i for i, f in zip(ids, folds["fold"], strict=True) if f != 1]
    std = Standardisation.fit(ds, train, pathway=PATHWAY)
    return ds, folds, std


def write_stan_data_inputs(root: Path, ds, folds: pd.DataFrame, std: Standardisation) -> None:
    """Write the case in the study's preprocessed-file layout for `stan_data.build(1)`."""
    pre = root / "results" / "02_preprocessing"
    (pre / "fold_01").mkdir(parents=True, exist_ok=True)
    (root / "results" / "01_splits").mkdir(parents=True, exist_ok=True)
    wide = ds.wide_observations()
    wide["patno"] = wide["participant_id"].astype(int)
    wide = wide.sort_values(["patno", "time_days"], kind="mergesort")
    wide["visit_index"] = wide.groupby("patno").cumcount()
    wide["dt_days_prev"] = wide.groupby("patno")["time_days"].diff()
    wide[["patno", "visit_index", "dt_days_prev", "updrs3", "moca", "scopa_aut", "ledd"]].to_csv(
        pre / "visits.csv", index=False
    )
    base = ds.participants.copy()
    base["patno"] = base["participant_id"].astype(int)
    base[["patno", "age_baseline_years", "sex", "genotype_assay_available"]].to_csv(
        pre / "baseline.csv", index=False
    )
    om = ds.marker_matrix(PATHWAY).reset_index().rename(columns={"participant_id": "patno"})
    om["patno"] = om["patno"].astype(int)
    om.to_csv(pre / "omics.csv", index=False)
    folds.to_csv(root / "results" / "01_splits" / "fold_assignments.csv", index=False)
    train = sorted(int(p) for p in folds.loc[folds["fold"] != 1, "patno"])
    digest = hashlib.sha256(",".join(str(p) for p in train).encode()).hexdigest()
    rows = [{"name": k, "mean": v[0], "sd": v[1], "train_patno_sha256": digest} for k, v in std.stats.items()]
    pd.DataFrame(rows).to_csv(pre / "fold_01" / "standardisation_stats_train.csv", index=False)


def simulate_case(seed: int = 202, n: int = 25):
    """Patterns, theta and rng seed for one replicate of the study simulator."""
    patterns = synthetic_visit_patterns(40, np.random.default_rng(seed))
    theta = draw_prior(np.random.default_rng(seed + 1), 2)
    return patterns, theta, seed + 2, n


def lp_case(seed: int = 303):
    """lp__ matrices for the chain-exclusion rule: clean, one trapped chain, two trapped chains."""
    rng = np.random.default_rng(seed)
    clean = rng.normal(-1000, 20, size=(300, 4))
    trapped = clean.copy()
    trapped[:, 2] -= 400
    two = rng.normal(-1000, 20, size=(300, 8))
    two[:, [1, 5]] -= 2000
    return {"clean": clean, "trapped": trapped, "two_trapped": two}


def fold_case(fold: int, seed: int = 404, n: int = 120, n_test: int = 30):
    """One synthetic held-out fold of one cohort: split data, metadata, test index, ages and draws.

    Folds 1 to 3 hold disjoint test participants of the same cohort. The twin draws are an
    oracle-like synthetic predictive, the true value plus noise, so every scoring path is exercised.
    """
    cohort = simulate_cohort(n, seed=seed)
    ds = cohort.dataset
    ids = ds.participant_ids
    part = ds.participants.set_index("participant_id").loc[ids]
    order = np.random.default_rng(seed + 1).permutation(n)
    tests = sorted(order[(fold - 1) * n_test : fold * n_test].tolist())
    rng = np.random.default_rng(seed + 100 + fold)
    sp = split_at_anchor(
        cohort.stan_data,
        tests,
        anchor_years=2.0,
        zero_dose_std=-1.5,
        cohort=part["cohort"].tolist(),
        sex=part["sex"].astype(int).tolist(),
    )
    y = sp.meta["y"].to_numpy(float)
    k = sp.meta["k"].to_numpy(int)
    upper = np.array([132.0, 30.0, 69.0])[k - 1]
    scale = np.array([4.0, 1.5, 3.0])[k - 1]
    draws = np.clip(
        y[None, :]
        + rng.normal(0, 1, size=(400, len(y))) * scale[None, :]
        + rng.normal(0, 1, size=(1, len(y))) * scale[None, :] * 0.5,
        0,
        upper[None, :],
    )
    age = part["age_baseline_years"].to_numpy(float)
    return sp, tests, age, draws


def write_fold_dir(root: Path, sp, tests, age, draws) -> Path:
    """Write a fold in the layout `score_fold.py --test` reads, and return the folder."""
    root.mkdir(parents=True, exist_ok=True)
    sp.fit.to_json(root / "stan_data_fit.json")
    sp.targets.to_json(root / "stan_data_targets.json")
    sp.meta.to_csv(root / "targets_meta.csv", index=False)
    (root / "test_index.json").write_text(json.dumps(sorted(tests)))
    (root / "test_age.json").write_text(json.dumps([float(a) for a in age]))
    record = {"fit_data_sha256": sp.fit.sha256(), "target_data_sha256": sp.targets.sha256()}
    (root / "data_record.json").write_text(json.dumps(record))
    (root / "score").mkdir(exist_ok=True)
    header = ["lp__"] + [f"y_rep_c.{q + 1}" for q in range(draws.shape[1])]
    with open(root / "score" / "gq_ho.csv", "w") as f:
        f.write("# synthetic draws for the golden test\n")
        f.write(",".join(header) + "\n")
        for row in draws:
            f.write(",".join(["0"] + [repr(float(v)) for v in row]) + "\n")
    return root
