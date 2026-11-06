"""Model registry, CmdStan location and compilation.

pd_v2 is registered at import. Its Stan files are checked against their sha256 when
`symphony_dt.models.pd_v2` is imported, and again by `get_model`. New models are registered
beside it with their own name and digest; the frozen files are never edited.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from symphony_dt.config import ModelConfig
from symphony_dt.errors import CmdStanNotFoundError, ModelIntegrityError
from symphony_dt.labels import validation

__all__ = [
    "ModelConfig",
    "ModelSpec",
    "cache_dir",
    "cmdstan_path",
    "cmdstan_version",
    "compile_model",
    "get_model",
    "list_models",
    "model_card",
    "register_model",
]

_REGISTRY: dict[str, ModelSpec] = {}


@validation("TESTED")
@dataclass(frozen=True)
class ModelSpec:
    """A registered model: its Stan programs, their digests and its model card.

    `programs` maps a role (`model`, `forecast_gq`, `conditional`) to a Stan file.
    `digests` maps the same roles to sha256 values, checked by `verify`.

    Validation: TESTED.
    """

    name: str
    programs: dict[str, Path]
    digests: dict[str, str]
    card: Path | None = None
    description: str = ""
    validation: str = "TESTED"
    default_config: ModelConfig = field(default_factory=ModelConfig.primary)

    def verify(self) -> None:
        """Check every program against its digest; raise ModelIntegrityError on a mismatch.

        Validation: TESTED.
        """
        for role, path in self.programs.items():
            expected = self.digests.get(role)
            if expected is None:
                continue
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if digest != expected:
                raise ModelIntegrityError(f"{self.name}/{role} ({path.name}) has sha256 {digest}, expected {expected}")

    def program(self, role: str = "model") -> Path:
        """Return the Stan file for `role`, or raise KeyError naming the roles.

        Validation: TESTED.
        """
        if role not in self.programs:
            raise KeyError(f"{self.name} has no program {role!r}; it has {sorted(self.programs)}")
        return self.programs[role]


@validation("TESTED")
def register_model(spec: ModelSpec, *, replace: bool = False) -> None:
    """Add a model to the registry. Registering an existing name raises unless `replace` is set.

    pd_v2 can never be replaced.

    Validation: TESTED.
    """
    if spec.name in _REGISTRY and not replace:
        raise ValueError(f"model {spec.name} is already registered")
    if spec.name == "pd_v2" and "pd_v2" in _REGISTRY:
        raise ValueError("pd_v2 is the frozen model of record and cannot be replaced")
    spec.verify()
    _REGISTRY[spec.name] = spec


@validation("TESTED")
def get_model(name: str = "pd_v2") -> ModelSpec:
    """Return a registered model after checking its digests.

    Validation: TESTED.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")
    spec = _REGISTRY[name]
    spec.verify()
    return spec


@validation("TESTED")
def list_models() -> list[str]:
    """Return the registered model names.

    Validation: TESTED.
    """
    return sorted(_REGISTRY)


@validation("TESTED")
def model_card(name: str = "pd_v2") -> str:
    """Return the model card of a registered model as Markdown text.

    Validation: TESTED.
    """
    spec = get_model(name)
    if spec.card is None:
        return f"# Model card: {name}\n\nNo card was registered.\n"
    return spec.card.read_text()


def _register_pd_v2() -> None:
    from symphony_dt.models import pd_v2

    register_model(
        ModelSpec(
            name="pd_v2",
            programs={
                "model": pd_v2.HERE / "model.stan",
                "forecast_gq": pd_v2.HERE / "forecast_gq.stan",
                "conditional": pd_v2.HERE / "conditional.stan",
            },
            digests={
                "model": pd_v2.MODEL_SHA256,
                "forecast_gq": pd_v2.FORECAST_GQ_SHA256,
                "conditional": pd_v2.CONDITIONAL_SHA256,
            },
            card=pd_v2.CARD,
            description="SYMPHONY v2 monotone latent state-space model of PD progression",
            validation="VALIDATED",
        )
    )


_register_pd_v2()


@validation("TESTED")
def cache_dir() -> Path:
    """Return the directory where compiled models are kept.

    `SYMPHONY_DT_CACHE` overrides the default `~/.cache/symphony-dt`.

    Validation: TESTED.
    """
    root = os.environ.get("SYMPHONY_DT_CACHE") or str(Path.home() / ".cache" / "symphony-dt")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


@validation("TESTED")
def cmdstan_path() -> Path:
    """Locate CmdStan: the `CMDSTAN` variable, then cmdstanpy's default, then `./.cmdstan`.

    Raises CmdStanNotFoundError with the install command when none is found.

    Validation: TESTED.
    """
    import cmdstanpy

    env = os.environ.get("CMDSTAN")
    if env and Path(env).is_dir():
        cmdstanpy.set_cmdstan_path(env)
        return Path(env)
    try:
        return Path(cmdstanpy.cmdstan_path())
    except (ValueError, RuntimeError):
        local = sorted(Path.cwd().glob(".cmdstan/cmdstan-*"))
        if local:
            cmdstanpy.set_cmdstan_path(str(local[-1]))
            return local[-1]
    raise CmdStanNotFoundError(
        "CmdStan was not found. Install it with: python -m cmdstanpy.install_cmdstan --version 2.39.0"
    )


@validation("TESTED")
def cmdstan_version() -> str | None:
    """Return the CmdStan version string, or None when CmdStan is not installed.

    Validation: TESTED.
    """
    try:
        path = cmdstan_path()
    except CmdStanNotFoundError:
        return None
    return path.name.replace("cmdstan-", "")


@validation("TESTED")
def compile_model(name: str = "pd_v2", program: str = "model", *, force: bool = False, threads: bool = True) -> Any:
    """Compile a registered program into the cache and return a `cmdstanpy.CmdStanModel`.

    The Stan file is copied into a cache folder named by its digest, so a compiled executable is
    reused across sessions and never written into the installed package.

    Validation: TESTED.
    """
    import cmdstanpy

    spec = get_model(name)
    source = spec.program(program)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cmdstan_path()
    folder = cache_dir() / f"{name}-{program}-{digest[:16]}"
    folder.mkdir(parents=True, exist_ok=True)
    stan_file = folder / source.name
    if not stan_file.exists() or hashlib.sha256(stan_file.read_bytes()).hexdigest() != digest:
        shutil.copyfile(source, stan_file)
    options = {"STAN_THREADS": True} if threads else None
    return cmdstanpy.CmdStanModel(stan_file=str(stan_file), cpp_options=options, force_compile=force)
