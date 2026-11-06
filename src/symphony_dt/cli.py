"""The `symphony-dt` command line.

Every command that writes files also writes a run manifest. Commands print aggregates only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from symphony_dt._version import __version__

app = typer.Typer(
    name="symphony-dt",
    help="Bayesian digital twins for longitudinal clinical cohorts.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

OutOpt = Annotated[Path, typer.Option("--out", "-o", help="Output folder.")]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(path: Path, mapping: str | None) -> Any:
    from symphony_dt.io import from_wide, read_dataset, read_table

    if path.is_dir():
        return read_dataset(path)
    frame = read_table(path)
    return from_wide(frame, mapping or "wide")


@app.command()
def about() -> None:
    """Print the logo, version, author, citation and documentation address."""
    from symphony_dt._about import about as _about

    _about()


@app.command()
def check(
    compile_models: Annotated[bool, typer.Option("--compile", help="Also compile pd_v2.")] = False,
) -> None:
    """Check the installation: version, frozen model digests and CmdStan."""
    from symphony_dt.models import cmdstan_version, compile_model, get_model

    spec = get_model("pd_v2")
    table = Table(title=f"symphony-dt {__version__}")
    table.add_column("check")
    table.add_column("result")
    for role, digest in sorted(spec.digests.items()):
        table.add_row(f"pd_v2 {role} sha256", f"{digest[:16]}... MATCH")
    version = cmdstan_version()
    table.add_row("CmdStan", version or "NOT FOUND (python -m cmdstanpy.install_cmdstan)")
    if compile_models and version:
        for program in ("model", "forecast_gq"):
            compile_model("pd_v2", program)
            table.add_row(f"compile {program}", "OK")
    console.print(table)
    if version is None:
        raise typer.Exit(code=1)


@app.command("validate-data")
def validate_data(
    path: Path,
    mapping: Annotated[str | None, typer.Option(help="Mapping for a table.")] = None,
    min_cell: int = 10,
) -> None:
    """Validate a dataset folder or a wide table and print aggregate counts."""
    from symphony_dt.errors import DataValidationError

    try:
        ds = _dataset(path, mapping)
    except DataValidationError as err:
        console.print(f"[red]INVALID[/red] {err}")
        raise typer.Exit(code=1) from err
    console.print_json(
        json.dumps({"valid": True, "fingerprint": ds.fingerprint(), **ds.summary(min_cell=min_cell)})
    )


@app.command()
def envelope(
    path: Path, mapping: Annotated[str | None, typer.Option(help="Mapping for a table.")] = None
) -> None:
    """Check a dataset against the pd_v2 validity envelope."""
    from symphony_dt.data import check_envelope

    report = check_envelope(_dataset(path, mapping))
    console.print_json(json.dumps(report.to_dict()))


@app.command()
def simulate(
    n: Annotated[int, typer.Option(help="Participants.")] = 100,
    seed: Annotated[int, typer.Option(help="Seed.")] = 20260729,
    out: OutOpt = Path("simulated"),
    fmt: Annotated[str, typer.Option("--format", help="parquet, csv or arrow.")] = "csv",
) -> None:
    """Simulate a synthetic cohort from the pd_v2 generative model."""
    from symphony_dt.data import simulate_cohort
    from symphony_dt.governance import RunManifest
    from symphony_dt.io import write_dataset

    cohort = simulate_cohort(n, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    write_dataset(cohort.dataset, out / "dataset", format=fmt)
    data_path = cohort.stan_data.to_json(out / "stan_data.json")
    truth = out / "truth.json"
    truth.write_text(json.dumps(cohort.truth(), indent=2, sort_keys=True) + "\n")
    manifest = RunManifest.create(
        "simulate",
        model="pd_v2",
        data_sha256=cohort.stan_data.sha256(),
        dataset_fingerprint=cohort.dataset.fingerprint(),
        seeds={"simulate": seed},
        settings={"n": n, "format": fmt},
        outputs={"sha256": {"stan_data.json": _sha(data_path), "truth.json": _sha(truth)}},
        command=["simulate", "--n", str(n), "--seed", str(seed), "--out", "{out}", "--format", fmt],
    )
    manifest.write(out / "manifest.json")
    console.print(f"simulated {n} participants into {out}; dataset {cohort.dataset.fingerprint()[:16]}")


@app.command()
def split(
    data: Annotated[Path, typer.Argument(help="stan_data.json or a dataset folder.")],
    out: OutOpt = Path("split"),
    test_fraction: float = 0.2,
    anchor_years: float = 2.0,
    seed: int = 20260729,
) -> None:
    """Hold out a random share of participants after an anchor, writing fit and target data."""
    import numpy as np

    from symphony_dt.data import StanData, build_stan_data
    from symphony_dt.forecast import split_at_anchor

    sd = StanData.from_json(data) if data.suffix == ".json" else build_stan_data(_dataset(data, None))
    rng = np.random.default_rng(seed)
    n = int(sd.data["N"])
    tests = sorted(rng.choice(n, size=max(1, int(round(test_fraction * n))), replace=False).tolist())
    holdout = split_at_anchor(sd, tests, anchor_years=anchor_years)
    holdout.write(out)
    console.print(f"{len(tests)} test participants, {len(holdout.meta)} targets, written to {out}")


@app.command()
def fit(
    data: Annotated[Path, typer.Argument(help="stan_data.json or a dataset folder.")],
    out: OutOpt = Path("fit"),
    chains: int = 4,
    warmup: int = 1000,
    draws: int = 1000,
    seed: int = 20260729,
    threads: int = 1,
    resume: bool = False,
    inits: Annotated[str, typer.Option(help="prior or random.")] = "prior",
) -> None:
    """Fit pd_v2 with CmdStan and print the gate verdict."""
    from symphony_dt.data import StanData, build_stan_data
    from symphony_dt.fit import fit as _fit

    sd = StanData.from_json(data) if data.suffix == ".json" else build_stan_data(_dataset(data, None))
    result = _fit(
        sd,
        chains=chains,
        iter_warmup=warmup,
        iter_sampling=draws,
        seed=seed,
        threads_per_chain=threads,
        output_dir=out,
        resume=resume,
        inits="random" if inits == "random" else "prior",
    )
    report = result.gates()
    result.manifest["gates"] = {k: v for k, v in report.to_dict().items() if k != "diagnostics"}
    result.save_manifest()
    for line in report.summary_lines():
        console.print(line)


@app.command()
def gates(fit_dir: Path, json_out: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Evaluate the convergence gates on a fit folder."""
    from symphony_dt.fit import load_fit

    report = load_fit(fit_dir).gates()
    if json_out:
        console.print_json(json.dumps(report.to_dict(), default=str))
    else:
        for line in report.summary_lines():
            console.print(line)
    if not report.converged:
        raise typer.Exit(code=2)


@app.command()
def forecast(
    fit_dir: Path, split_dir: Path, out: OutOpt = Path("forecast"), keep: int = 400, seed: int = 20261021
) -> None:
    """Forecast held-out targets with the generated-quantities program (exact route)."""
    import pandas as pd

    from symphony_dt.data import StanData
    from symphony_dt.fit import load_fit
    from symphony_dt.forecast import HoldoutSplit, forecast_exact

    split_obj = HoldoutSplit(
        fit=StanData.from_json(split_dir / "stan_data_fit.json"),
        targets=StanData.from_json(split_dir / "stan_data_targets.json"),
        meta=pd.read_csv(split_dir / "targets_meta.csv"),
        test_index=json.loads((split_dir / "test_index.json").read_text()),
        anchor_years=2.0,
    )
    fc = forecast_exact(load_fit(fit_dir), split_obj, total_keep=keep, seed=seed, output_dir=out)
    fc.to_csv(out / "forecast_quantiles.csv")
    console.print(f"{fc.n_draws} draws for {len(fc.targets)} targets; quantiles in {out}")


@app.command()
def evaluate(
    split_dir: Path,
    gq_csv: Path,
    out: OutOpt = Path("scores"),
    spec: Annotated[Path | None, typer.Option(help="Frozen specification JSON.")] = None,
) -> None:
    """Score a held-out forecast against LOCF and the LMM under a frozen specification."""
    import pandas as pd

    from symphony_dt.data import StanData
    from symphony_dt.evaluate import ScoringSpec, freeze_spec, load_frozen_spec, score_fold
    from symphony_dt.forecast import read_gq_csv

    if spec is None:
        out.mkdir(parents=True, exist_ok=True)
        spec = out / "frozen_spec.json"
        freeze_spec(ScoringSpec.default(), spec)
        console.print(f"froze the default specification to {spec}")
    scoring, frozen = load_frozen_spec(spec)
    fit_data = StanData.from_json(split_dir / "stan_data_fit.json").data
    targets = StanData.from_json(split_dir / "stan_data_targets.json").data
    meta = pd.read_csv(split_dir / "targets_meta.csv")
    tests = json.loads((split_dir / "test_index.json").read_text())
    draws = read_gq_csv(gq_csv, int(targets["n_obs"]))
    scores = score_fold(fit_data, targets, meta, draws, tests, spec=scoring, frozen=frozen)
    scores.write(out)
    for line in scores.lines:
        console.print(line)


@app.command()
def sbc(
    replicates: int = 20,
    n: int = 40,
    chains: int = 4,
    warmup: int = 200,
    draws: int = 200,
    seed: int = 0,
    method: str = "bonferroni",
    out: OutOpt = Path("sbc"),
) -> None:
    """Run a small prior-regime SBC campaign for pd_v2 (slow: one fit per replicate)."""
    from symphony_dt.calibration import pd_v2_prior_design, run_sbc

    design = pd_v2_prior_design(
        n, chains=chains, iter_warmup=warmup, iter_sampling=draws, output_root=str(out / "fits")
    )
    result = run_sbc(design, n_replicates=replicates, seed=seed, method=method)  # type: ignore[arg-type]
    out.mkdir(parents=True, exist_ok=True)
    result.ranks.to_csv(out / "ranks.csv", index=False)
    (out / "sbc_report.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    v = result.verdict
    console.print(
        f"{v.m} parameters, method {v.method}: passed {v.passed}; rejected {v.rejected}; "
        f"uncorrected family-wise risk {v.fwer_uncorrected:.3f}"
    )


@app.command()
def ppc(fit_dir: Path, out: OutOpt = Path("ppc"), keep: int = 400, seed: int = 1) -> None:
    """Posterior predictive check of a fit on its own data: coverage, boundaries and PIT."""
    from symphony_dt.calibration import ppc_summary
    from symphony_dt.data import StanData
    from symphony_dt.fit import load_fit
    from symphony_dt.forecast import read_gq_csv, run_standalone_gq, thin_draws
    from symphony_dt.models import compile_model

    result = load_fit(fit_dir)
    if result.stan_data_path is None:
        raise typer.BadParameter("the fit folder holds no stan_data.json")
    out.mkdir(parents=True, exist_ok=True)
    thin = thin_draws(result.csv_files, out / "thin.csv", total_keep=keep)
    exe = compile_model("pd_v2", "forecast_gq")
    gq_csv = run_standalone_gq(exe.exe_file, result.stan_data_path, thin, out / "gq.csv", seed=seed)
    data = StanData.from_json(result.stan_data_path).data
    draws = read_gq_csv(gq_csv, int(data["n_obs"]))
    summary = ppc_summary(draws, data)
    (out / "ppc_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    console.print_json(json.dumps(summary))


@app.command()
def report(
    manifests: list[Path], out: OutOpt = Path("report.html"), title: str = "symphony-dt report"
) -> None:
    """Write an HTML report from one or more run manifests."""
    from symphony_dt.report import write_html_report

    loaded = [json.loads(p.read_text()) for p in manifests]
    gate = next((m.get("gates") for m in loaded if m.get("gates")), None)
    write_html_report(out, loaded, title=title, gates=gate)
    console.print(f"wrote {out}")


@app.command()
def reproduce(manifest: Path, workdir: Annotated[Path | None, typer.Option()] = None) -> None:
    """Rerun the command recorded in a manifest and compare output digests."""
    from symphony_dt.governance import reproduce as _reproduce

    result = _reproduce(manifest, workdir=workdir)
    for name, row in result.compared.items():
        console.print(f"{name}: {'MATCH' if row['match'] else 'DIFFER'}")
    console.print(f"REPRODUCED {result.reproduced}")
    if not result.reproduced:
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point of the `symphony-dt` script."""
    app(prog_name="symphony-dt")


if __name__ == "__main__":  # pragma: no cover
    main()
