"""Run manifests: what ran, on which data fingerprint, with which code, seeds and verdicts.

A manifest never holds data. It holds digests of the data, the model and the environment, so a
reader can check that two runs used the same inputs without seeing them.
"""

from __future__ import annotations

import hashlib
import html
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from symphony_dt._version import __version__
from symphony_dt.labels import validation

__all__ = ["RunManifest", "environment", "fingerprint"]

_PACKAGES = ("numpy", "pandas", "scipy", "cmdstanpy", "arviz", "pydantic", "matplotlib")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


@validation("TESTED")
def fingerprint(obj: Any) -> str:
    """Return the sha256 of a canonical JSON serialisation of `obj`.

    Dictionaries are sorted by key and numpy values are converted, so equal content gives an equal
    digest. Objects with a `fingerprint()` method, such as TwinDataset, use it.

    Validation: TESTED.
    """
    method = getattr(obj, "fingerprint", None)
    if callable(method):
        return str(method())
    payload = json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


@validation("TESTED")
def environment() -> dict[str, Any]:
    """Return the computing environment: Python, platform, package versions and CmdStan.

    Validation: TESTED.
    """
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        from symphony_dt.models import cmdstan_version

        cmdstan = cmdstan_version()
    except Exception:  # pragma: no cover - CmdStan lookup must never break a manifest
        cmdstan = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "cmdstan": cmdstan,
    }


@validation("TESTED")
@dataclass
class RunManifest:
    """The record of one fit, forecast, evaluation or simulation.

    Validation: TESTED.
    """

    kind: str
    package_version: str
    created_utc: str
    model: str | None = None
    model_sha256: str | None = None
    data_sha256: str | None = None
    dataset_fingerprint: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    seeds: dict[str, int] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] | None = None
    overrides: list[dict[str, Any]] = field(default_factory=list)
    specification: dict[str, Any] | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    command: list[str] | None = None
    wall_seconds: float | None = None

    @classmethod
    def create(cls, kind: str, *, started: float | None = None, **fields: Any) -> RunManifest:
        """Create a manifest stamped with the package version, the time and the environment.

        Validation: TESTED.
        """
        wall = None if started is None else round(time.time() - started, 3)
        return cls(
            kind=kind,
            package_version=__version__,
            created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            environment=environment(),
            wall_seconds=wall,
            **fields,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a JSON-serialisable dict.

        Validation: TESTED.
        """
        return dict(_jsonable(asdict(self)))

    def sha256(self) -> str:
        """Return the digest of the manifest content.

        Validation: TESTED.
        """
        return fingerprint(self.to_dict())

    def write(self, path: str | Path) -> Path:
        """Write the manifest as indented JSON and return the path.

        Validation: TESTED.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return target

    @classmethod
    def read(cls, path: str | Path) -> RunManifest:
        """Read a manifest written by `write`.

        Validation: TESTED.
        """
        raw = json.loads(Path(path).read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def to_html(self) -> str:
        """Return a self-contained HTML page describing the run.

        Validation: TESTED.
        """

        def rows(d: dict[str, Any]) -> str:
            return "".join(
                f"<tr><th>{html.escape(str(k))}</th><td><code>{html.escape(json.dumps(v, default=str))}"
                "</code></td></tr>"
                for k, v in d.items()
            )

        body = self.to_dict()
        main = {
            k: body[k]
            for k in (
                "kind",
                "package_version",
                "created_utc",
                "model",
                "model_sha256",
                "data_sha256",
                "dataset_fingerprint",
                "wall_seconds",
            )
        }
        sections = [
            ("Run", main),
            ("Seeds", body["seeds"]),
            ("Settings", body["settings"]),
            ("Environment", body["environment"]),
        ]
        if body["gates"]:
            sections.append(("Gates", {k: v for k, v in body["gates"].items() if k != "diagnostics"}))
        if body["overrides"]:
            sections.append(("Overrides", {str(i): o for i, o in enumerate(body["overrides"])}))
        if body["specification"]:
            sections.append(("Specification", body["specification"]))
        parts = "".join(f"<h2>{t}</h2><table>{rows(d)}</table>" for t, d in sections)
        return (
            "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>symphony-dt run "
            "manifest</title><style>"
            "body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto}"
            "th{text-align:left;padding-right:1rem;vertical-align:top}code{white-space:pre-wrap}</style>"
            f"</head><body><h1>Run manifest: {html.escape(self.kind)}</h1>{parts}</body></html>\n"
        )
