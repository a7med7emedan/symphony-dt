from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import yaml


@dataclass(frozen=True)
class RunConfig:
    project: str
    seed: int
    outputs_root: str
    overwrite: bool = False


@dataclass(frozen=True)
class IOConfig:
    input_dir: str
    cohort: Literal["PDLIKE", "ALLCOHORTS"]
    baseline_filename_pdlike: str
    clinic_filename_pdlike: str
    baseline_filename_all: str
    clinic_filename_all: str
    omics_filename: str


@dataclass(frozen=True)
class CohortConfig:
    require_sex: bool = True
    require_age_baseline_years: bool = True
    pd_like_only: bool = True


@dataclass(frozen=True)
class DemoConfig:
    enabled: bool = True
    n_patients: int = 25
    patient_id_strategy: Literal["sorted_first", "stable_hash", "clinical_best"] = "sorted_first"


@dataclass(frozen=True)
class CVConfig:
    n_folds: int = 5


@dataclass(frozen=True)
class ShardingConfig:
    method: Literal["hash", "stride"] = "hash"


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(frozen=True)
class AppConfig:
    run: RunConfig
    io: IOConfig
    cohort: CohortConfig
    demo: DemoConfig
    cv: CVConfig
    sharding: ShardingConfig
    logging: LoggingConfig


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required key '{key}' in {ctx}")
    return d[key]


def _as_int(x: Any, ctx: str) -> int:
    if isinstance(x, bool):
        raise ValueError(f"Invalid int (bool) in {ctx}")
    try:
        return int(x)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Invalid int in {ctx}: {x!r}") from e


def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping at top level")

    run_d = _require(raw, "run", "root")
    io_d = _require(raw, "io", "root")
    cohort_d = raw.get("cohort", {}) or {}
    demo_d = raw.get("demo", {}) or {}
    cv_d = raw.get("cv", {}) or {}
    sh_d = raw.get("sharding", {}) or {}
    log_d = raw.get("logging", {}) or {}

    run = RunConfig(
        project=str(_require(run_d, "project", "run")),
        seed=_as_int(_require(run_d, "seed", "run"), "run.seed"),
        outputs_root=str(_require(run_d, "outputs_root", "run")),
        overwrite=bool(run_d.get("overwrite", False)),
    )

    cohort = str(_require(io_d, "cohort", "io")).upper()
    if cohort not in {"PDLIKE", "ALLCOHORTS"}:
        raise ValueError("io.cohort must be one of: PDLIKE, ALLCOHORTS")

    io_cfg = IOConfig(
        input_dir=str(_require(io_d, "input_dir", "io")),
        cohort=cohort,  # type: ignore[assignment]
        baseline_filename_pdlike=str(_require(io_d, "baseline_filename_pdlike", "io")),
        clinic_filename_pdlike=str(_require(io_d, "clinic_filename_pdlike", "io")),
        baseline_filename_all=str(_require(io_d, "baseline_filename_all", "io")),
        clinic_filename_all=str(_require(io_d, "clinic_filename_all", "io")),
        omics_filename=str(_require(io_d, "omics_filename", "io")),
    )

    cohort_cfg = CohortConfig(
        require_sex=bool(cohort_d.get("require_sex", True)),
        require_age_baseline_years=bool(cohort_d.get("require_age_baseline_years", True)),
        pd_like_only=bool(cohort_d.get("pd_like_only", True)),
    )

    demo_cfg = DemoConfig(
        enabled=bool(demo_d.get("enabled", True)),
        n_patients=_as_int(demo_d.get("n_patients", 25), "demo.n_patients"),
        patient_id_strategy=str(demo_d.get("patient_id_strategy", "sorted_first")),
    )
    if demo_cfg.n_patients <= 0:
        raise ValueError("demo.n_patients must be > 0")
    if demo_cfg.patient_id_strategy not in {"sorted_first", "stable_hash", "clinical_best"}:
        raise ValueError("demo.patient_id_strategy must be one of: sorted_first, stable_hash, clinical_best")

    cv_cfg = CVConfig(n_folds=_as_int(cv_d.get("n_folds", 5), "cv.n_folds"))
    if cv_cfg.n_folds < 2:
        raise ValueError("cv.n_folds must be >= 2")

    sharding_method = str(sh_d.get("method", "hash")).lower()
    if sharding_method not in {"hash", "stride"}:
        raise ValueError("sharding.method must be one of: hash, stride")
    sharding_cfg = ShardingConfig(method=sharding_method)  # type: ignore[arg-type]

    logging_cfg = LoggingConfig(level=str(log_d.get("level", "INFO")).upper())

    return AppConfig(
        run=run,
        io=io_cfg,
        cohort=cohort_cfg,
        demo=demo_cfg,
        cv=cv_cfg,
        sharding=sharding_cfg,
        logging=logging_cfg,
    )
