"""cov90_* is the canonical coverage name; cov95_* stays readable as a deprecated alias for one release."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from symphony_dt.aggregate.all import aggregate_all_results
from symphony_dt.cli import app
from symphony_dt.evaluation.metrics import (
    COVERAGE_COLUMNS,
    DEPRECATED_COVERAGE_COLUMNS,
    add_deprecated_coverage_aliases,
    interval_coverage_by_patient,
)
from symphony_dt.evaluation.stratified import evaluation_stratified_coverage_table
from symphony_dt.evaluation.tables import evaluation_coverage_table
from symphony_dt.forecast.io import write_table_atomic
from symphony_dt.forecast.schema_extra import validate_evaluation_coverage
from symphony_dt.notebook.ingest import ingest_notebook
from symphony_dt.parity import parity_check_run_dir
from symphony_dt.schemas.contracts import ColumnRule, TableContract, validate_table

runner = CliRunner()


def _pred() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2],
            "ledd_anchor": [0.0, 100.0, 0.0, 50.0],
            "y_target_updrs3": [12.0, 30.0, 10.0, 11.0],
            "y_target_moca": [27.0, 27.0, 28.0, np.nan],
            "y_target_scopa": [6.0, 6.0, 5.0, 5.0],
            "pred_p05_updrs3": [5.0, 5.0, 5.0, 5.0],
            "pred_p95_updrs3": [20.0, 20.0, 20.0, 20.0],
            "pred_p05_moca": [26.0, 26.0, 26.0, 26.0],
            "pred_p95_moca": [30.0, 30.0, 30.0, 30.0],
            "pred_p05_scopa": [0.0, 0.0, 0.0, 0.0],
            "pred_p95_scopa": [10.0, 10.0, 10.0, 10.0],
        }
    )


def _coverage_row(prefix: str, value: float) -> dict:
    row = {"scope": "one_step", "stratum_type": "", "stratum_value": "", "n_patients": 2}
    row.update({f"{prefix}_{s}": value for s in ("updrs3", "moca", "scopa", "overall")})
    return row


def test_per_patient_coverage_writes_cov90_and_identical_cov95_aliases() -> None:
    cov = interval_coverage_by_patient(_pred())
    assert set(COVERAGE_COLUMNS) <= set(cov.columns)
    assert set(DEPRECATED_COVERAGE_COLUMNS) <= set(cov.columns)
    for new, old in zip(COVERAGE_COLUMNS, DEPRECATED_COVERAGE_COLUMNS):
        pd.testing.assert_series_equal(cov[new], cov[old], check_names=False)
    assert cov.loc[cov["patient_id"] == 1, "cov90_updrs3"].item() == pytest.approx(0.5)
    # Aliases come after the canonical columns.
    cols = list(cov.columns)
    assert max(cols.index(c) for c in COVERAGE_COLUMNS) < min(cols.index(c) for c in DEPRECATED_COVERAGE_COLUMNS)


def test_summary_and_stratified_tables_carry_both_names() -> None:
    t = evaluation_coverage_table(_pred(), scope="one_step")
    assert t["cov90_overall"].item() == pytest.approx(t["cov95_overall"].item())
    baseline = pd.DataFrame({"patient_id": [1, 2], "sex": ["M", "F"], "age_baseline_years": [60.0, 55.0]})
    clinic = pd.DataFrame({"patient_id": [1, 1, 2, 2], "visit_id": [10, 11, 20, 21], "t_days_since_baseline": [0, 100, 0, 120]})
    s = evaluation_stratified_coverage_table(_pred(), baseline, clinic, scope="one_step")
    assert set(COVERAGE_COLUMNS) <= set(s.columns)
    np.testing.assert_allclose(s["cov90_overall"], s["cov95_overall"])


def test_old_tables_with_only_cov95_are_upgraded() -> None:
    old = pd.DataFrame([_coverage_row("cov95", 0.8)])
    up = add_deprecated_coverage_aliases(old)
    assert up["cov90_moca"].item() == pytest.approx(0.8)
    assert up["cov95_moca"].item() == pytest.approx(0.8)


@pytest.mark.parametrize("prefixes", [("cov90",), ("cov95",), ("cov90", "cov95")])
def test_evaluation_coverage_contract_accepts_either_name(prefixes) -> None:
    row: dict = {}
    for p in prefixes:
        row.update(_coverage_row(p, 0.9))
    out = validate_evaluation_coverage(pd.DataFrame([row]))
    assert out["cov90_overall"].item() == pytest.approx(0.9)


def test_evaluation_coverage_contract_rejects_missing_and_out_of_range() -> None:
    with pytest.raises(ValueError, match="cov90_overall"):
        validate_evaluation_coverage(pd.DataFrame([{"scope": "one_step", "n_patients": 1}]))
    with pytest.raises(ValueError):
        validate_evaluation_coverage(pd.DataFrame([_coverage_row("cov95", 1.5)]))


def test_column_rule_aliases_are_generic() -> None:
    c = TableContract(table_name="t", required=(ColumnRule("new", float, aliases=("old",)),))
    out = validate_table(pd.DataFrame({"old": [1.0]}), c)
    assert out["new"].tolist() == [1.0]


def test_aggregate_all_merges_old_and_new_coverage_tables(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_table_atomic(runs / "r0" / "results" / "evaluation_coverage.csv", pd.DataFrame([_coverage_row("cov95", 0.8)]))
    new = add_deprecated_coverage_aliases(pd.DataFrame([_coverage_row("cov90", 0.6)]))
    write_table_atomic(runs / "r1" / "results" / "evaluation_coverage.csv", new)
    aggregate_all_results(runs_root=runs, out_dir=tmp_path / "out")
    out = pd.read_csv(tmp_path / "out" / "evaluation_coverage.csv")
    assert out["cov90_overall"].item() == pytest.approx(0.7)
    assert out["cov95_overall"].item() == pytest.approx(0.7)


def test_quickstart_run_writes_cov90_and_parity_accepts_old_tables(tmp_path: Path) -> None:
    qs = tmp_path / "qs"
    assert runner.invoke(app, ["quickstart", "--workdir", str(qs), "--n-patients", "20"]).exit_code == 0
    run_dir = next((qs / "outputs").glob("symphony-dt__*__demo"))
    cov_path = run_dir / "results" / "evaluation_coverage.csv"
    cov = pd.read_csv(cov_path)
    assert set(COVERAGE_COLUMNS) <= set(cov.columns)
    np.testing.assert_allclose(cov["cov90_overall"], cov["cov95_overall"], equal_nan=True)
    res = parity_check_run_dir(run_dir)
    assert res.passed, (res.missing_files, res.errors)
    assert res.warnings == []

    # A table from the previous release (cov95_* only) still passes, with a warning.
    cov.drop(columns=list(COVERAGE_COLUMNS)).to_csv(cov_path, index=False)
    res = parity_check_run_dir(run_dir)
    assert res.passed, res.errors
    assert any("cov95" in w for w in res.warnings)


def test_notebook_ingest_flags_deprecated_coverage_columns(tmp_path: Path) -> None:
    nb = {"cells": [{"cell_type": "code", "source": ["df['cov95_overall'].mean()\n", "df.cov90_moca\n"]}]}
    p = tmp_path / "x.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    info = ingest_notebook(p)
    refs = {r["column"]: r for r in info.coverage_column_refs}
    assert refs["cov95_overall"]["deprecated"] and refs["cov95_overall"]["canonical"] == "cov90_overall"
    assert not refs["cov90_moca"]["deprecated"]
