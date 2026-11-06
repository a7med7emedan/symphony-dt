"""Reproduce a run from its manifest and compare the output digests."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from symphony_dt.governance.manifest import RunManifest
from symphony_dt.labels import validation

__all__ = ["ReproduceReport", "reproduce"]


@validation("TESTED")
@dataclass
class ReproduceReport:
    """The outcome of `reproduce`: per-output digest comparison.

    Validation: TESTED.
    """

    manifest: str
    workdir: str
    compared: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def reproduced(self) -> bool:
        """True when every recorded output digest was reproduced."""
        return bool(self.compared) and all(v["match"] for v in self.compared.values())


@validation("TESTED")
def reproduce(manifest_path: str | Path, *, workdir: str | Path | None = None) -> ReproduceReport:
    """Rerun the CLI command recorded in a manifest in a fresh folder and compare output digests.

    The manifest must hold `command` (the CLI arguments, with `{out}` standing for the output
    folder) and `outputs["sha256"]` (file name to digest). Runs whose inputs are not reachable
    from the new folder cannot be reproduced and raise FileNotFoundError.

    Validation: TESTED.
    """
    from typer.testing import CliRunner

    from symphony_dt.cli import app

    manifest = RunManifest.read(manifest_path)
    if not manifest.command:
        raise ValueError(f"{manifest_path} records no command, so it cannot be rerun")
    expected = manifest.outputs.get("sha256", {})
    if not expected:
        raise ValueError(f"{manifest_path} records no output digests")
    work = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="sdt-reproduce-"))
    work.mkdir(parents=True, exist_ok=True)
    args = [a.replace("{out}", str(work)) for a in manifest.command]
    result = CliRunner().invoke(app, args)
    if result.exit_code != 0:
        raise RuntimeError(f"the recorded command failed: {result.output.strip()[-500:]}")
    report = ReproduceReport(manifest=str(manifest_path), workdir=str(work))
    for name, digest in expected.items():
        path = work / name
        if not path.exists():
            report.compared[name] = {"expected": digest, "found": None, "match": False}
            continue
        found = hashlib.sha256(path.read_bytes()).hexdigest()
        report.compared[name] = {"expected": digest, "found": found, "match": found == digest}
    return report
