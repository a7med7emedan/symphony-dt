"""Run anywhere: portable bundles, Slurm job arrays, container recipes, JSON Schemas and a JSON twin service.

* `write_bundle` packs a run's inputs, settings, manifests and an environment lock with a SHA256SUMS file, so the
  run can be repeated on another machine; `verify_bundle` checks every digest there.
* `slurm_script` writes an sbatch job array with one CmdStan chain per task and a dependent job that judges the
  gates once every chain has finished.
* `container_recipe` returns a Dockerfile or an Apptainer definition with CmdStan installed.
* `json_schema` returns JSON Schemas for the study file, the dataset tables and a twin visit, so R, Julia or any
  language can produce valid inputs.
* `serve_twin` runs a JSON-lines loop on standard streams: any program sends a command and reads a reply.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import IO, Any, Literal

from symphony_dt._version import __version__
from symphony_dt.labels import validation

__all__ = ["container_recipe", "json_schema", "serve_twin", "slurm_script", "verify_bundle", "write_bundle"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@validation("TESTED")
def write_bundle(
    files: Iterable[str | Path], output: str | Path, *, root: str | Path | None = None, note: str = ""
) -> Path:
    """Pack files into a `.tar.gz` run bundle with SHA256SUMS, an environment lock and a README.

    Paths inside the bundle are relative to `root` (default: the common parent). Never include participant-level
    data unless the receiving machine is governed for it.

    Validation: TESTED.
    """
    paths = [Path(f).resolve() for f in files]
    if not paths:
        raise ValueError("a bundle needs at least one file")
    if root is not None:
        base = Path(root).resolve()
    else:
        common = Path(os.path.commonpath([str(p) for p in paths]))
        base = common.parent if common in paths else common
    sums = "".join(f"{_sha(p)}  {p.relative_to(base)}\n" for p in paths)
    try:
        lock = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:  # pragma: no cover
        lock = ""
    readme = (
        f"symphony-dt run bundle\nsymphony-dt {__version__}, Python {platform.python_version()}, "
        f"{platform.platform()}\n{note}\nVerify with: symphony-dt bundle verify <this file>\n"
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for p in paths:
            tar.add(p, arcname=str(Path("bundle") / p.relative_to(base)))
        for name, text in (("SHA256SUMS", sums), ("requirements.lock", lock), ("README.txt", readme)):
            data = text.encode()
            info = tarfile.TarInfo(f"bundle/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out


@validation("TESTED")
def verify_bundle(bundle: str | Path, *, extract_to: str | Path | None = None) -> dict[str, bool]:
    """Extract a bundle and check every file against SHA256SUMS; return file to match.

    Validation: TESTED.
    """
    import tempfile

    target = Path(extract_to) if extract_to else Path(tempfile.mkdtemp(prefix="sdt-bundle-"))
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(target, filter="data")
    root = target / "bundle"
    result = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        result[name] = (root / name).exists() and _sha(root / name) == digest
    return result


@validation("TESTED")
def slurm_script(
    data: str | Path,
    output_dir: str | Path,
    *,
    chains: int = 4,
    warmup: int = 1000,
    draws: int = 1000,
    seed: int = 20260729,
    threads_per_chain: int = 1,
    time: str = "24:00:00",
    mem: str = "16G",
    partition: str | None = None,
    account: str | None = None,
    python: str = "python",
    cmdstan: str | None = None,
    modules: Sequence[str] = (),
) -> str:
    """Return an sbatch job-array script: one chain per array task, then a dependent gates job.

    Chains share one output folder and seed and differ by chain id, so the result equals a local run with the
    same settings. Submit with `sbatch fit.sbatch`; the script submits the gates job itself.

    Validation: TESTED.
    """
    extra = "".join(f"#SBATCH --{k}={v}\n" for k, v in (("partition", partition), ("account", account)) if v)
    mods = "".join(f"module load {m}\n" for m in modules)
    env = f"export CMDSTAN={cmdstan}\n" if cmdstan else ""
    return f"""#!/bin/bash
#SBATCH --job-name=symphony-fit
#SBATCH --array=1-{chains}
#SBATCH --cpus-per-task={threads_per_chain}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={output_dir}/slurm-%A_%a.out
{extra}set -euo pipefail
{mods}{env}export STAN_NUM_THREADS={threads_per_chain}
mkdir -p {output_dir}
{python} -m symphony_dt fit {data} --out {output_dir} --chain-ids "$SLURM_ARRAY_TASK_ID" --warmup {warmup} \\
  --draws {draws} --seed {seed} --threads {threads_per_chain} --resume
if [ "$SLURM_ARRAY_TASK_ID" = "1" ]; then
  sbatch --dependency=afterok:$SLURM_ARRAY_JOB_ID --job-name=symphony-gates --time=01:00:00 --mem=8G \\
    --output={output_dir}/slurm-gates-%j.out --wrap "{python} -m symphony_dt gates {output_dir} --json"
fi
"""


_DOCKER = """# symphony-dt with CmdStan. Build: docker build -t symphony-dt .
# Run: docker run --rm -v "$PWD":/work symphony-dt about
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential make curl \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "symphony-dt[io]=={version}"
RUN python -m cmdstanpy.install_cmdstan --version 2.39.0 --dir /opt/cmdstan --cores 4
ENV CMDSTAN=/opt/cmdstan/cmdstan-2.39.0 SYMPHONY_DT_CACHE=/opt/symphony-cache
RUN symphony-dt check --compile
WORKDIR /work
ENTRYPOINT ["symphony-dt"]
"""

_APPTAINER = """# symphony-dt with CmdStan for HPC. Build: apptainer build symphony-dt.sif symphony-dt.def
Bootstrap: docker
From: python:3.12-slim

%post
    apt-get update && apt-get install -y --no-install-recommends build-essential make curl
    rm -rf /var/lib/apt/lists/*
    pip install --no-cache-dir "symphony-dt[io]=={version}"
    python -m cmdstanpy.install_cmdstan --version 2.39.0 --dir /opt/cmdstan --cores 4
    CMDSTAN=/opt/cmdstan/cmdstan-2.39.0 SYMPHONY_DT_CACHE=/opt/symphony-cache symphony-dt check --compile

%environment
    export CMDSTAN=/opt/cmdstan/cmdstan-2.39.0
    export SYMPHONY_DT_CACHE=/opt/symphony-cache

%runscript
    exec symphony-dt "$@"
"""


@validation("TESTED")
def container_recipe(kind: Literal["docker", "apptainer"] = "docker", version: str | None = None) -> str:
    """Return a Dockerfile or an Apptainer definition that installs symphony-dt and CmdStan 2.39.0.

    Validation: TESTED.
    """
    template = {"docker": _DOCKER, "apptainer": _APPTAINER}[kind]
    return template.replace("{version}", version or __version__)


@validation("TESTED")
def json_schema(name: Literal["study", "dataset", "visit"]) -> dict[str, Any]:
    """Return a JSON Schema: the study file, the dataset tables, or one twin visit.

    Validation: TESTED.
    """
    if name == "study":
        from symphony_dt.study import StudyConfig

        schema = StudyConfig.model_json_schema()
        schema["title"] = "symphony-dt study file"
        return schema
    if name == "visit":
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "symphony-dt twin visit",
            "type": "object",
            "required": ["time_days", "scores"],
            "properties": {
                "time_days": {"type": "integer", "minimum": 0},
                "scores": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "propertyNames": {"enum": ["updrs3", "moca", "scopa_aut"]},
                },
                "ledd_std": {"type": ["number", "null"]},
            },
        }
    row = {"participant_id": {"type": "string"}, "time_days": {"type": "integer", "minimum": 0}}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "symphony-dt dataset tables",
        "type": "object",
        "required": ["participants", "observations"],
        "properties": {
            "participants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["participant_id", "cohort"],
                    "properties": {
                        "participant_id": {"type": "string"},
                        "cohort": {"type": "string"},
                        "age_baseline_years": {"type": "number"},
                        "sex": {"enum": [0, 1]},
                    },
                },
            },
            "observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [*row, "endpoint", "value"],
                    "properties": {**row, "endpoint": {"type": "string"}, "value": {"type": "number"}},
                },
            },
            "visits": {
                "type": "array",
                "items": {"type": "object", "required": list(row), "properties": {**row, "ledd": {"type": "number"}}},
            },
            "markers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["participant_id", "name", "value"],
                    "properties": {
                        "participant_id": {"type": "string"},
                        "name": {"type": "string"},
                        "value": {"type": "number"},
                    },
                },
            },
        },
    }


@validation("EXPERIMENTAL")
def serve_twin(twin_path: str | Path, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> int:
    """Run a JSON-lines command loop for one saved twin; return the number of commands handled.

    Commands, one JSON object per line: `{"op": "observe", "visit": {...}}`, `{"op": "forecast", "horizons":
    [1, 2]}`, `{"op": "next_visit", "candidates": [0.5, 1], "target_years": 2}`, `{"op": "milestone",
    "endpoint": "moca", "threshold": 26}`, `{"op": "status"}`, `{"op": "save"}` and `{"op": "quit"}`. Each reply
    is one JSON object with `ok` and either `result` or `error`. The twin is saved after every observe.

    Validation: EXPERIMENTAL.
    """
    from symphony_dt.twin import ParticipantTwin

    src, dst = stdin or sys.stdin, stdout or sys.stdout
    twin = ParticipantTwin.load(twin_path)
    handled = 0
    for line in src:
        if not line.strip():
            continue
        handled += 1
        try:
            cmd = json.loads(line)
            op = cmd.get("op")
            if op == "quit":
                dst.write(json.dumps({"ok": True, "result": "bye"}) + "\n")
                break
            if op == "observe":
                entry = twin.observe(cmd["visit"])
                twin.save(twin_path)
                result: Any = {k: entry[k] for k in ("time_days", "particle_ess_median", "draw_ess")}
            elif op == "forecast":
                fc = twin.forecast(cmd.get("horizons", [1.0, 2.0]), n_samples=int(cmd.get("n_samples", 1000)))
                result = {
                    "quantiles": fc.quantiles().to_dict(orient="records"),
                    "uncertainty": fc.metadata["uncertainty"],
                }
            elif op == "next_visit":
                result = twin.next_visit(
                    cmd.get("candidates", [0.25, 0.5, 1.0, 1.5]),
                    target_years=float(cmd.get("target_years", 2.0)),
                    endpoint=cmd.get("endpoint", "updrs3"),
                ).to_dict(orient="records")
            elif op == "milestone":
                result = twin.milestone_curve(
                    cmd["endpoint"], float(cmd["threshold"]), years=float(cmd.get("years", 5.0))
                ).to_dict(orient="records")
            elif op == "status":
                result = twin.status()
            elif op == "save":
                result = str(twin.save(twin_path))
            else:
                raise ValueError(f"unknown op {op!r}")
            dst.write(json.dumps({"ok": True, "op": op, "result": result}, default=float) + "\n")
        except Exception as err:  # every error becomes a reply, never a crash of the loop
            dst.write(json.dumps({"ok": False, "error": f"{type(err).__name__}: {err}"}) + "\n")
        dst.flush()
    return handled
