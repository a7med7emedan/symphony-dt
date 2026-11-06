"""One study file, one command: `symphony-dt run study.yaml`.

A `StudyConfig` names the data, the eligibility rules, the model arm, the held-out design, the sampler, any gate
override, the forecast, the scoring specification and the outputs. `run_study` runs the stages in order, writes
each stage's outputs and a manifest into its own folder, and skips a stage that is already complete for the same
configuration. `dry_run` stops before fitting, so a configuration and its data can be checked in seconds.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from symphony_dt import constants as C
from symphony_dt.labels import validation

__all__ = ["STAGES", "StudyConfig", "StudyResult", "init_study", "load_study", "run_study"]

#: Stages in the order they run.
STAGES: tuple[str, ...] = (
    "data",
    "eligibility",
    "profile",
    "build",
    "split",
    "fit",
    "gates",
    "forecast",
    "evaluate",
    "report",
)
_DRY = ("data", "eligibility", "profile", "build", "split")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


@validation("TESTED")
class DataSection(_Section):
    """Where the data come from: a dataset folder, a table with a mapping, or a simulation.

    Validation: TESTED.
    """

    path: str | None = None
    mapping: str | None = None
    simulate_n: int | None = None
    simulate_seed: int = C.MASTER_SEED


@validation("TESTED")
class ModelSection(_Section):
    """The registered model, its arm and its pathway markers.

    Validation: TESTED.
    """

    name: str = "pd_v2"
    arm: Literal["primary", "a01_sensitivity"] = "primary"
    pathway: list[str] = Field(default_factory=list)


@validation("TESTED")
class SplitSection(_Section):
    """The held-out design: a random share of participants masked after the anchor.

    Validation: TESTED.
    """

    test_fraction: float = 0.2
    anchor_years: float = 2.0
    seed: int = C.MASTER_SEED


@validation("TESTED")
class SamplerSection(_Section):
    """CmdStan NUTS settings.

    Validation: TESTED.
    """

    chains: int = 4
    warmup: int = 1000
    draws: int = 1000
    seed: int = C.MASTER_SEED
    threads_per_chain: int = 1
    adapt_delta: float = 0.8
    max_treedepth: int = 10
    inits: Literal["prior", "random"] = "prior"


@validation("TESTED")
class OverrideSection(_Section):
    """An explicit gate override; it needs a reason and an author and is logged.

    Validation: TESTED.
    """

    reason: str
    author: str
    c2_max_rhat: float | None = None
    c3_min_ess_bulk: float | None = None
    min_retained_chains: int | None = None
    trap_multiple: float | None = None


@validation("TESTED")
class ForecastSection(_Section):
    """Exact-route forecast settings.

    Validation: TESTED.
    """

    total_keep: int = 400
    seed: int = 20261021


@validation("TESTED")
class OutputSection(_Section):
    """Table formats, figure formats and whether to draw figures.

    Validation: TESTED.
    """

    tables: list[str] = Field(default_factory=lambda: ["csv", "parquet", "xlsx"])
    figures: list[str] = Field(default_factory=lambda: ["png", "svg"])
    plots: bool = True
    min_cell: int = 10


@validation("TESTED")
class StudyConfig(_Section):
    """The whole study in one object, loaded from `study.yaml`.

    Validation: TESTED.
    """

    name: str = "study"
    output_dir: str = "symphony_study"
    data: DataSection = Field(default_factory=DataSection)
    eligibility: list[dict[str, Any]] = Field(default_factory=list)
    model: ModelSection = Field(default_factory=ModelSection)
    split: SplitSection = Field(default_factory=SplitSection)
    sampler: SamplerSection = Field(default_factory=SamplerSection)
    gate_override: OverrideSection | None = None
    forecast: ForecastSection = Field(default_factory=ForecastSection)
    scoring: dict[str, Any] = Field(default_factory=dict)
    outputs: OutputSection = Field(default_factory=OutputSection)

    def digest(self, *sections: str) -> str:
        """Return a digest of the named sections, used to decide whether a stage must rerun.

        Validation: TESTED.
        """
        from symphony_dt.governance.manifest import fingerprint

        dump = self.model_dump()
        return fingerprint({k: dump[k] for k in sections} if sections else dump)


_TEMPLATE = """# symphony-dt study file. Run with: symphony-dt run study.yaml   (add --dry-run to stop before fitting)
name: my-study
output_dir: symphony_study

data:
  # Either a dataset folder written by write_dataset, or a table plus a mapping (wide, ppmi, omop, fhir, sdtm,
  # redcap, generic_long or your own YAML file), or a simulation for a dry run with no data at all.
  path: null
  mapping: null
  simulate_n: 120
  simulate_seed: 20260729

eligibility:            # applied in order; see the Eligibility guide
  - {rule: cohort_in, cohorts: [PD, Prodromal]}
  - {rule: age_range, low: 30, high: 95}
  - {rule: required_at_baseline}
  - {rule: min_visits, n: 2}

model: {name: pd_v2, arm: primary, pathway: []}
split: {test_fraction: 0.2, anchor_years: 2.0, seed: 20260729}
sampler: {chains: 4, warmup: 1000, draws: 1000, seed: 20260729, threads_per_chain: 1, adapt_delta: 0.8,
          max_treedepth: 10, inits: prior}
gate_override: null     # or {reason: "...", author: "...", c2_max_rhat: 1.02}; it is logged
forecast: {total_keep: 400, seed: 20261021}
scoring: {}             # overrides of the default ScoringSpec, for example {min_bin: 20}; recorded as a deviation
outputs: {tables: [csv, parquet, xlsx], figures: [png, svg], plots: true, min_cell: 10}
"""


@validation("TESTED")
def init_study(directory: str | Path) -> Path:
    """Write a commented `study.yaml` template into `directory` and return its path. An existing file is kept.

    Validation: TESTED.
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "study.yaml"
    if target.exists():
        return target
    target.write_text(_TEMPLATE)
    return target


@validation("TESTED")
def load_study(path: str | Path) -> StudyConfig:
    """Read and validate a study file. Relative data paths resolve against the file's folder.

    Validation: TESTED.
    """
    file = Path(path)
    raw = yaml.safe_load(file.read_text()) or {}
    config = StudyConfig(**raw)
    if config.data.path and not Path(config.data.path).is_absolute():
        config.data.path = str((file.parent / config.data.path).resolve())
    if not Path(config.output_dir).is_absolute():
        config.output_dir = str((file.parent / config.output_dir).resolve())
    return config


@validation("TESTED")
class StudyResult:
    """What `run_study` produced: stage folders, statuses and the objects of the last run.

    Validation: TESTED.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.status: dict[str, str] = {}
        self.objects: dict[str, Any] = {}

    def summary(self) -> dict[str, str]:
        """Return the status of every stage.

        Validation: TESTED.
        """
        return dict(self.status)


def _stage_dir(root: Path, name: str) -> Path:
    path = root / f"{STAGES.index(name):02d}_{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _done(folder: Path, key: str) -> bool:
    marker = folder / ".done"
    return marker.exists() and marker.read_text().strip() == key


def _mark(folder: Path, key: str, manifest: dict[str, Any]) -> None:
    from symphony_dt.governance.manifest import RunManifest

    RunManifest.create(**manifest).write(folder / "manifest.json")
    (folder / ".done").write_text(key + "\n")


@validation("TESTED")
def run_study(
    config: StudyConfig | str | Path,
    *,
    stages: Sequence[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> StudyResult:
    """Run the study, stage by stage.

    `stages` restricts the run to named stages (their inputs must exist from an earlier run). `dry_run` runs
    data, eligibility, profile, build and split only. A stage whose marker matches the current configuration is
    skipped unless `force` is set; its outputs are reloaded from disk.

    Validation: TESTED.
    """
    from symphony_dt.config import ModelConfig
    from symphony_dt.data import (
        EligibilityCriteria,
        Standardisation,
        StanData,
        apply_eligibility,
        build_stan_data,
        check_envelope,
        profile,
        simulate_cohort,
    )
    from symphony_dt.forecast import HoldoutSplit, split_at_anchor
    from symphony_dt.io import read_dataset, write_dataset
    from symphony_dt.outputs import write_tables

    cfg = config if isinstance(config, StudyConfig) else load_study(config)
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "study.resolved.json").write_text(json.dumps(cfg.model_dump(), indent=2, default=str) + "\n")
    wanted = list(_DRY if dry_run else (stages or STAGES))
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stages {unknown}; known: {list(STAGES)}")
    result = StudyResult(root)
    objects = result.objects
    sections = {
        "data": ("data",),
        "eligibility": ("data", "eligibility"),
        "profile": ("data", "eligibility"),
        "build": ("data", "eligibility", "model"),
        "split": ("data", "eligibility", "model", "split"),
        "fit": ("data", "eligibility", "model", "split", "sampler"),
        "gates": ("data", "eligibility", "model", "split", "sampler", "gate_override"),
        "forecast": ("data", "eligibility", "model", "split", "sampler", "forecast"),
        "evaluate": ("data", "eligibility", "model", "split", "sampler", "forecast", "scoring"),
        "report": ("data", "eligibility", "model", "split", "sampler", "forecast", "scoring", "outputs"),
    }

    def run(name: str) -> bool:
        if name not in wanted:
            return False
        folder = _stage_dir(root, name)
        key = cfg.digest(*sections[name])
        if not force and _done(folder, key):
            result.status[name] = "skipped (complete)"
            return False
        result.status[name] = "ran"
        return True

    def manifest(kind: str, **extra: Any) -> dict[str, Any]:
        return {"kind": f"study:{kind}", "model": cfg.model.name, "settings": {"study": cfg.name}, **extra}

    # data -----------------------------------------------------------------------------------------
    data_dir = root / "00_data"
    if run("data"):
        started = time.time()
        if cfg.data.simulate_n:
            ds = simulate_cohort(
                cfg.data.simulate_n, seed=cfg.data.simulate_seed, n_pathway=len(cfg.model.pathway)
            ).dataset
        elif cfg.data.path and Path(cfg.data.path).is_dir():
            ds = read_dataset(cfg.data.path)
        elif cfg.data.path:
            from symphony_dt.io import from_wide, read_table

            ds = from_wide(read_table(cfg.data.path), cfg.data.mapping or "wide")
        else:
            raise ValueError("the study file names no data: set data.path or data.simulate_n")
        write_dataset(ds, data_dir / "dataset", format="parquet")
        (data_dir / "envelope.json").write_text(json.dumps(check_envelope(ds).to_dict(), indent=2) + "\n")
        _mark(
            data_dir,
            cfg.digest(*sections["data"]),
            manifest("data", dataset_fingerprint=ds.fingerprint(), started=started),
        )
    if (data_dir / "dataset").exists():
        objects["dataset"] = read_dataset(data_dir / "dataset")

    # eligibility ----------------------------------------------------------------------------------
    elig_dir = root / "01_eligibility"
    if run("eligibility"):
        crit = EligibilityCriteria.from_list(cfg.eligibility)
        res = apply_eligibility(objects["dataset"], crit)
        write_dataset(res.eligible, elig_dir / "dataset", format="parquet")
        res.attrition_table(min_cell=cfg.outputs.min_cell).to_csv(elig_dir / "attrition.csv", index=False)
        (elig_dir / "attrition.md").write_text(res.to_markdown(min_cell=cfg.outputs.min_cell))
        objects["eligibility"] = res
        if cfg.outputs.plots:
            from symphony_dt.plots import plot_attrition, save_figure

            ax = plot_attrition(res, min_cell=cfg.outputs.min_cell)
            save_figure(ax.figure, elig_dir / "attrition", cfg.outputs.figures)
        _mark(
            elig_dir,
            cfg.digest(*sections["eligibility"]),
            manifest(
                "eligibility",
                dataset_fingerprint=res.eligible.fingerprint(),
                outputs={"eligible": res.n_eligible},
            ),
        )
    eligible_any = read_dataset(elig_dir / "dataset") if (elig_dir / "dataset").exists() else objects.get("dataset")
    if eligible_any is None:
        raise ValueError("no dataset: run the data stage first")
    eligible: Any = eligible_any
    objects["eligible"] = eligible

    # profile --------------------------------------------------------------------------------------
    prof_dir = root / "02_profile"
    if run("profile"):
        prof = profile(eligible, min_cell=cfg.outputs.min_cell)
        (prof_dir / "profile.md").write_text(prof.to_markdown())
        write_tables(
            {"profile_endpoints": prof.endpoints, "profile_cohorts": prof.cohorts.astype(str)},
            prof_dir,
            [f for f in cfg.outputs.tables if f != "parquet"],
        )
        if cfg.outputs.plots:
            from symphony_dt.plots import save_figure

            save_figure(prof.plot(), prof_dir / "profile", cfg.outputs.figures)
        objects["profile"] = prof
        _mark(prof_dir, cfg.digest(*sections["profile"]), manifest("profile"))

    # build ----------------------------------------------------------------------------------------
    build_dir = root / "03_build"
    model_cfg = ModelConfig.primary() if cfg.model.arm == "primary" else ModelConfig.a01_sensitivity()
    rng = np.random.default_rng(cfg.split.seed)
    if run("build"):
        sd = build_stan_data(
            eligible,
            config=model_cfg,
            standardisation=Standardisation.fit(eligible, pathway=cfg.model.pathway),
            pathway=cfg.model.pathway,
        )
        sd.to_json(build_dir / "stan_data.json")
        (build_dir / "participants.json").write_text(json.dumps(sd.participant_ids) + "\n")
        (build_dir / "audit.json").write_text(json.dumps(sd.audit, indent=2, default=str) + "\n")
        _mark(build_dir, cfg.digest(*sections["build"]), manifest("build", data_sha256=sd.sha256()))
    if (build_dir / "stan_data.json").exists():
        objects["stan_data"] = StanData.from_json(build_dir / "stan_data.json")
        objects["participant_ids"] = json.loads((build_dir / "participants.json").read_text())

    # split ----------------------------------------------------------------------------------------
    split_dir = root / "04_split"
    if run("split"):
        sd = objects["stan_data"]
        n = int(sd.data["N"])
        tests = sorted(rng.choice(n, size=max(1, round(cfg.split.test_fraction * n)), replace=False).tolist())
        part = eligible.participants.set_index("participant_id").loc[objects["participant_ids"]]
        split = split_at_anchor(
            sd,
            tests,
            anchor_years=cfg.split.anchor_years,
            cohort=part["cohort"].tolist(),
            sex=part["sex"].astype(int).tolist() if "sex" in part.columns else None,
        )
        split.write(split_dir)
        (split_dir / "age.json").write_text(json.dumps(part["age_baseline_years"].astype(float).tolist()) + "\n")
        _mark(
            split_dir,
            cfg.digest(*sections["split"]),
            manifest("split", outputs={"targets": len(split.meta), "test_participants": len(tests)}),
        )
    if (split_dir / "stan_data_fit.json").exists():
        objects["split"] = HoldoutSplit(
            fit=StanData.from_json(split_dir / "stan_data_fit.json"),
            targets=StanData.from_json(split_dir / "stan_data_targets.json"),
            meta=pd.read_csv(split_dir / "targets_meta.csv"),
            test_index=json.loads((split_dir / "test_index.json").read_text()),
            anchor_years=cfg.split.anchor_years,
        )
    if dry_run:
        return result

    # fit ------------------------------------------------------------------------------------------
    fit_dir = root / "05_fit"
    if run("fit"):
        from symphony_dt.fit import fit

        s = cfg.sampler
        objects["fit"] = fit(
            objects["split"].fit,
            chains=s.chains,
            iter_warmup=s.warmup,
            iter_sampling=s.draws,
            seed=s.seed,
            threads_per_chain=s.threads_per_chain,
            adapt_delta=s.adapt_delta,
            max_treedepth=s.max_treedepth,
            inits=s.inits,
            output_dir=fit_dir / "run",
            resume=True,
        )
        _mark(fit_dir, cfg.digest(*sections["fit"]), manifest("fit", settings=s.model_dump()))
    if (fit_dir / "run" / "chains").exists() and "fit" not in objects:
        from symphony_dt.fit import load_fit

        objects["fit"] = load_fit(fit_dir / "run")

    # gates ----------------------------------------------------------------------------------------
    gates_dir = root / "06_gates"
    if run("gates"):
        from symphony_dt.diagnostics import GateOverride
        from symphony_dt.outputs import posterior_summary

        ov = GateOverride(**cfg.gate_override.model_dump()) if cfg.gate_override else None
        report = objects["fit"].gates(override=ov)
        (gates_dir / "gates.json").write_text(json.dumps(report.to_dict(), indent=2, default=str) + "\n")
        (gates_dir / "gates.txt").write_text("\n".join(report.summary_lines()) + "\n")
        write_tables({"posterior_summary": posterior_summary(objects["fit"])}, gates_dir, cfg.outputs.tables)
        if cfg.outputs.plots:
            from symphony_dt.plots import plot_energy, plot_rank, plot_trace, save_figure

            worst = [k for k, _ in report.lowest_ess(4)]
            files = objects["fit"].csv_files
            save_figure(plot_trace(files, worst)[0].figure, gates_dir / "trace_lowest_ess", cfg.outputs.figures)
            save_figure(plot_rank(files, worst)[0].figure, gates_dir / "rank_lowest_ess", cfg.outputs.figures)
            save_figure(plot_energy(files).figure, gates_dir / "energy", cfg.outputs.figures)
        objects["gates"] = report
        _mark(
            gates_dir,
            cfg.digest(*sections["gates"]),
            manifest(
                "gates",
                gates={k: v for k, v in report.to_dict().items() if k != "diagnostics"},
                overrides=[report.override] if report.override else [],
            ),
        )

    # forecast -------------------------------------------------------------------------------------
    fc_dir = root / "07_forecast"
    if run("forecast"):
        from symphony_dt.forecast import forecast_exact
        from symphony_dt.outputs import forecast_table

        fc = forecast_exact(
            objects["fit"],
            objects["split"],
            total_keep=cfg.forecast.total_keep,
            seed=cfg.forecast.seed,
            output_dir=fc_dir / "gq",
        )
        write_tables({"forecasts": forecast_table(fc, objects["participant_ids"])}, fc_dir, cfg.outputs.tables)
        np.save(fc_dir / "draws.npy", fc.draws)
        objects["forecast"] = fc
        _mark(fc_dir, cfg.digest(*sections["forecast"]), manifest("forecast"))
    if (fc_dir / "draws.npy").exists() and "forecast" not in objects:
        from symphony_dt.forecast import Forecast

        objects["forecast"] = Forecast(np.load(fc_dir / "draws.npy"), objects["split"].meta)

    # evaluate -------------------------------------------------------------------------------------
    ev_dir = root / "08_evaluate"
    if run("evaluate"):
        from symphony_dt.evaluate import DeviationRecord, ScoringSpec, freeze_spec, score_fold
        from symphony_dt.outputs import scores_table

        default = ScoringSpec.default()
        frozen = freeze_spec(default, ev_dir / "frozen_spec.json")
        spec = default.model_copy(update=cfg.scoring) if cfg.scoring else default
        deviation = (
            DeviationRecord("study file overrides the default scoring specification", "study file", dict(cfg.scoring))
            if cfg.scoring
            else None
        )
        sp = objects["split"]
        age = json.loads((split_dir / "age.json").read_text())
        scores = score_fold(
            sp.fit.data,
            sp.targets.data,
            sp.meta,
            objects["forecast"].draws,
            sp.test_index,
            spec=spec,
            frozen=frozen,
            deviation=deviation,
            age=age,
        )
        (ev_dir / "scores.json").write_text(scores.to_json())
        (ev_dir / "scores.txt").write_text("\n".join(scores.lines) + "\n")
        write_tables({"scores": scores_table(scores.scores)}, ev_dir, cfg.outputs.tables)
        if cfg.outputs.plots:
            from symphony_dt.plots import plot_coverage_by_window, plot_crps_differences, save_figure

            save_figure(plot_coverage_by_window(scores.scores).figure, ev_dir / "coverage", cfg.outputs.figures)
            save_figure(plot_crps_differences(scores.scores).figure, ev_dir / "crps_differences", cfg.outputs.figures)
        objects["scores"] = scores
        _mark(
            ev_dir,
            cfg.digest(*sections["evaluate"]),
            manifest("evaluate", specification=scores.specification),
        )

    # report ---------------------------------------------------------------------------------------
    rep_dir = root / "09_report"
    if run("report"):
        from symphony_dt.report import write_html_report

        manifests = [json.loads(p.read_text()) for p in sorted(root.glob("*/manifest.json"))]
        gates = json.loads((gates_dir / "gates.json").read_text()) if (gates_dir / "gates.json").exists() else None
        score_cells = json.loads((ev_dir / "scores.json").read_text()) if (ev_dir / "scores.json").exists() else None
        write_html_report(
            rep_dir / "report.html",
            manifests,
            title=f"symphony-dt study: {cfg.name}",
            gates=gates,
            scores=score_cells,
            notes=["Associational forecasts. No causal or treatment claim."],
        )
        _mark(rep_dir, cfg.digest(*sections["report"]), manifest("report"))
    return result
