from pathlib import Path

from symphony_dt.parity import parity_check_run_dir


def test_parity_check_flags_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    res = parity_check_run_dir(run_dir)
    assert not res.passed
    assert len(res.missing_files) > 0
