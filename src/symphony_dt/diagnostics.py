"""Convergence gates of the SYMPHONY v2 study, fixed in one place.

C2: max rank-normalised split R-hat at most 1.01 over the global parameters. C3: min bulk ESS at
least 400. At least four retained chains. A chain is excluded when its mean `lp__` falls more than
5.0 within-chain standard deviations below the best chain. `GateReport.converged` derives from these
three verdicts and nothing else. Divergences, tree depth saturation, E-BFMI and tail ESS are
reported beside it.

The thresholds are module constants. A different threshold needs an explicit `GateOverride` with a
reason and an author; the override is written into the report, the manifest and a warning.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from symphony_dt.errors import GateError, SymphonyWarning
from symphony_dt.labels import validation

__all__ = [
    "C2_MAX_RHAT",
    "C3_MIN_ESS_BULK",
    "MIN_RETAINED_CHAINS",
    "PER_PARTICIPANT_BLOCKS",
    "TRAP_MULTIPLE",
    "GateOverride",
    "GateOverrideWarning",
    "GateReport",
    "assess",
    "chain_csvs_in_order",
    "chain_id",
    "e_bfmi",
    "is_global",
    "read_sampler_columns",
    "retained_chains",
    "summary_on_chains",
]

#: C2. Maximum rank-normalised split R-hat over the global parameters.
C2_MAX_RHAT: Final[float] = 1.01
#: C3. Minimum bulk effective sample size over the global parameters.
C3_MIN_ESS_BULK: Final[float] = 400.0
#: Fewer retained chains than this is a failed fit.
MIN_RETAINED_CHAINS: Final[int] = 4
#: A chain is trapped when its mean lp__ is more than this many sd_within below the best chain.
TRAP_MULTIPLE: Final[float] = 5.0
#: Per-participant blocks, excluded from the gates.
PER_PARTICIPANT_BLOCKS: Final[tuple[str, ...]] = ("eps", "s0", "b_raw", "bmat")


@validation("TESTED")
class GateOverrideWarning(SymphonyWarning):
    """Issued whenever a gate threshold is overridden.

    Validation: TESTED.
    """


@validation("TESTED")
@dataclass(frozen=True)
class GateOverride:
    """An explicit, logged change to one or more gate thresholds.

    A reason and an author are required. Every use issues a GateOverrideWarning and is recorded in
    the gate report and in the run manifest, so no threshold changes silently.

    Validation: TESTED.
    """

    reason: str
    author: str
    c2_max_rhat: float | None = None
    c3_min_ess_bulk: float | None = None
    min_retained_chains: int | None = None
    trap_multiple: float | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip() or not self.author.strip():
            raise GateError("a GateOverride needs a reason and an author")
        if all(
            v is None for v in (self.c2_max_rhat, self.c3_min_ess_bulk, self.min_retained_chains, self.trap_multiple)
        ):
            raise GateError("a GateOverride must change at least one threshold")

    def thresholds(self) -> dict[str, float]:
        """Return the thresholds in force under this override.

        Validation: TESTED.
        """
        return {
            "c2_max_rhat": C2_MAX_RHAT if self.c2_max_rhat is None else self.c2_max_rhat,
            "c3_min_ess_bulk": C3_MIN_ESS_BULK if self.c3_min_ess_bulk is None else self.c3_min_ess_bulk,
            "min_retained_chains": float(
                MIN_RETAINED_CHAINS if self.min_retained_chains is None else self.min_retained_chains
            ),
            "trap_multiple": TRAP_MULTIPLE if self.trap_multiple is None else self.trap_multiple,
        }

    def record(self) -> dict[str, Any]:
        """Return the override as a JSON-serialisable record.

        Validation: TESTED.
        """
        return {"reason": self.reason, "author": self.author, "thresholds": self.thresholds()}


def _thresholds(override: GateOverride | None) -> dict[str, float]:
    if override is None:
        return {
            "c2_max_rhat": C2_MAX_RHAT,
            "c3_min_ess_bulk": C3_MIN_ESS_BULK,
            "min_retained_chains": float(MIN_RETAINED_CHAINS),
            "trap_multiple": TRAP_MULTIPLE,
        }
    warnings.warn(
        f"gate thresholds overridden by {override.author}: {override.reason}",
        GateOverrideWarning,
        stacklevel=3,
    )
    return override.thresholds()


@validation("VALIDATED")
def is_global(name: str) -> bool:
    """Return True for a global parameter, False for sampler columns and per-participant blocks.

    Validation: VALIDATED.
    """
    if name.endswith("__"):
        return False
    for block in PER_PARTICIPANT_BLOCKS:
        if name.startswith(block + "[") or name.startswith(block + ".") or name == block:
            return False
    return True


@validation("VALIDATED")
def retained_chains(lp: ArrayLike, *, override: GateOverride | None = None) -> dict[str, Any]:
    """Apply the pre-declared chain-exclusion rule to `lp`, shaped (draws, chains).

    Returns the decision and every number behind it, so a reader can apply another multiple to the
    same run without refitting.

    Validation: VALIDATED. Equal to the study's `sbc.retained_chains`.
    """
    th = _thresholds(override)
    multiple = th["trap_multiple"]
    minimum = int(th["min_retained_chains"])
    lp = np.asarray(lp, float)
    if lp.ndim != 2 or lp.shape[1] < 1:
        raise GateError(f"lp must be (draws, chains), got shape {lp.shape}")
    per_chain_mean = lp.mean(axis=0)
    per_chain_sd = lp.std(axis=0, ddof=1)
    sd_within = float(per_chain_sd.mean())
    best = float(np.max(per_chain_mean))
    cut = best - multiple * sd_within
    keep = [int(c) for c in range(per_chain_mean.size) if per_chain_mean[c] >= cut]
    n_chains = int(per_chain_mean.size)
    return {
        "rule": (
            f"a chain is trapped, and excluded, if its mean lp__ falls more than "
            f"{multiple} x sd_within below the highest chain mean, where sd_within "
            f"is the pooled within-chain sd of lp__ in this same run"
        ),
        "pre_declared": override is None,
        "trap_multiple": multiple,
        "sd_within": sd_within,
        "cut_lp": cut,
        "best_chain_mean_lp": best,
        "per_chain_mean_lp": [float(v) for v in per_chain_mean],
        "per_chain_sd_lp": [float(v) for v in per_chain_sd],
        "retained_chains": [c + 1 for c in keep],
        "excluded_chains": [c + 1 for c in range(n_chains) if c not in keep],
        "n_retained": len(keep),
        "n_chains": n_chains,
        "trapping_rate": 1.0 - len(keep) / n_chains,
        "meets_min_retained": len(keep) >= minimum,
        "min_retained_chains_required": minimum,
        "keep_index": keep,
    }


_HEADER_ID = re.compile(r"#\s*id\s*=\s*(\d+)")


@validation("TESTED")
def chain_id(csv_file: str | Path) -> int:
    """Return the chain id of a CmdStan CSV file.

    The id is the trailing `_N` of the file name. A one-chain run has no such suffix, because cmdstanpy names its
    file without one, so the `id = N` line of the CmdStan header is read instead.

    Validation: TESTED.
    """
    path = Path(csv_file)
    head, sep, tail = path.stem.rpartition("_")
    if sep and head and tail.isdigit():
        return int(tail)
    with open(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            match = _HEADER_ID.match(line)
            if match:
                return int(match.group(1))
    raise ValueError(f"{path}: no chain id in the file name or the CmdStan header")


@validation("VALIDATED")
def chain_csvs_in_order(csv_files: Sequence[str | Path]) -> list[str]:
    """Sort chain CSV files by their chain id, never lexically.

    Validation: VALIDATED. The id comes from `chain_id`, which also reads one-chain files.
    """
    return sorted((str(p) for p in csv_files), key=chain_id)


@validation("VALIDATED")
def summary_on_chains(csv_files: Sequence[str | Path], keep: Sequence[int] | None = None) -> pd.DataFrame:
    """Run CmdStan's own `stansummary` on a subset of chains and return its table.

    The statistics are not re-implemented: rank-normalised split R-hat, bulk and tail ESS come from
    the same binary `fit.summary()` calls, restricted to the retained chains.

    Validation: VALIDATED.
    """
    from symphony_dt.models import cmdstan_path

    chosen = list(range(len(csv_files))) if keep is None else list(keep)
    selected = [str(csv_files[i]) for i in chosen]
    binary = cmdstan_path() / "bin" / "stansummary"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "summary.csv"
        subprocess.run([str(binary), "-s", "9", "-c", str(out), *selected], check=True, capture_output=True, text=True)
        frame = pd.read_csv(out, index_col=0, comment="#")
    frame.index = [str(i).strip('"') for i in frame.index]
    return frame


@validation("TESTED")
def read_sampler_columns(csv_files: Sequence[str | Path]) -> dict[str, np.ndarray]:
    """Read lp__, divergent__, treedepth__ and energy__ from chain CSVs as (draws, chains) arrays.

    Validation: TESTED.
    """
    wanted = ["lp__", "divergent__", "treedepth__", "energy__"]
    frames = [pd.read_csv(p, comment="#", usecols=lambda c: c in wanted) for p in csv_files]
    lengths = {len(f) for f in frames}
    if len(lengths) != 1:
        raise GateError(f"unequal draws per chain {sorted(lengths)}")
    return {
        c: np.column_stack([f[c].to_numpy(float) for f in frames])
        for c in wanted
        if all(c in f.columns for f in frames)
    }


@validation("TESTED")
def e_bfmi(energy: ArrayLike) -> list[float]:
    """Return the energy Bayesian fraction of missing information per chain.

    `energy` is (draws, chains). Values below 0.3 suggest poor exploration.

    Validation: TESTED.
    """
    e = np.asarray(energy, float)
    num = np.sum(np.diff(e, axis=0) ** 2, axis=0)
    den = np.sum((e - e.mean(axis=0)) ** 2, axis=0)
    return [float(v) for v in num / den]


@validation("VALIDATED")
@dataclass
class GateReport:
    """The verdict of the convergence gates on one fit.

    `converged` is derived from `meets_c2`, `meets_c3` and `meets_min_retained` only.

    Validation: VALIDATED design. The derivation matches the study's `combine_fold`.
    """

    max_rhat: float
    min_ess_bulk: float
    min_ess_tail: float
    chain_exclusion: dict[str, Any]
    thresholds: dict[str, float]
    divergences: int = 0
    max_treedepth_hits: int = 0
    max_treedepth: int = 10
    e_bfmi: list[float] = field(default_factory=list)
    diagnostics: dict[str, dict[str, float]] = field(default_factory=dict)
    override: dict[str, Any] | None = None
    n_draws_per_chain: int = 0

    @property
    def meets_c2(self) -> bool:
        """C2: max R-hat at most the threshold."""
        return bool(self.max_rhat <= self.thresholds["c2_max_rhat"])

    @property
    def meets_c3(self) -> bool:
        """C3: min bulk ESS at least the threshold."""
        return bool(self.min_ess_bulk >= self.thresholds["c3_min_ess_bulk"])

    @property
    def meets_min_retained(self) -> bool:
        """At least the required number of retained chains."""
        return bool(self.chain_exclusion["n_retained"] >= self.thresholds["min_retained_chains"])

    @property
    def converged(self) -> bool:
        """True only when C2, C3 and the retained-chain count all pass."""
        return self.meets_c2 and self.meets_c3 and self.meets_min_retained

    def lowest_ess(self, n: int = 5) -> list[tuple[str, float]]:
        """Return the `n` global parameters with the lowest bulk ESS.

        Validation: TESTED.
        """
        ranked = sorted(self.diagnostics.items(), key=lambda kv: kv[1]["ess_bulk"])[:n]
        return [(k, v["ess_bulk"]) for k, v in ranked]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record, in the layout of the study's fit record.

        Validation: TESTED.
        """
        return {
            "chains": self.chain_exclusion["n_chains"],
            "draws_per_chain": self.n_draws_per_chain,
            "chain_exclusion": self.chain_exclusion,
            "max_rhat": self.max_rhat,
            "min_ess_bulk": self.min_ess_bulk,
            "min_ess_tail": self.min_ess_tail,
            "divergences": self.divergences,
            "max_treedepth_hits": self.max_treedepth_hits,
            "e_bfmi": self.e_bfmi,
            "meets_c2": self.meets_c2,
            "meets_c3": self.meets_c3,
            "meets_min_retained": self.meets_min_retained,
            "converged": self.converged,
            "thresholds": self.thresholds,
            "override": self.override,
            "diagnostics": self.diagnostics,
        }

    def summary_lines(self) -> list[str]:
        """Return a short human-readable verdict.

        Validation: TESTED.
        """
        ce = self.chain_exclusion
        return [
            f"chains retained {ce['n_retained']} of {ce['n_chains']} (minimum "
            f"{int(self.thresholds['min_retained_chains'])})",
            f"C2 max R-hat {self.max_rhat:.5f} (<= {self.thresholds['c2_max_rhat']}): "
            f"{'PASS' if self.meets_c2 else 'FAIL'}",
            f"C3 min bulk ESS {self.min_ess_bulk:.1f} (>= {self.thresholds['c3_min_ess_bulk']}): "
            f"{'PASS' if self.meets_c3 else 'FAIL'}",
            f"min tail ESS {self.min_ess_tail:.1f}; divergences {self.divergences}; "
            f"tree depth hits {self.max_treedepth_hits}",
            f"CONVERGED {self.converged}" + ("  (thresholds overridden)" if self.override else ""),
        ]


@validation("VALIDATED")
def assess_tables(
    lp: ArrayLike,
    summary: pd.DataFrame,
    *,
    override: GateOverride | None = None,
    sampler: dict[str, np.ndarray] | None = None,
    max_treedepth: int = 10,
) -> GateReport:
    """Build a GateReport from an lp__ matrix and a stansummary table of the retained chains.

    This is the computational core of `assess`, separated so it can be tested without CmdStan.

    Validation: VALIDATED.
    """
    th = _thresholds(override)
    lp_arr = np.asarray(lp, float)
    trap = retained_chains(lp_arr, override=override)
    diagnostics: dict[str, dict[str, float]] = {}
    for name in summary.index:
        if is_global(str(name)):
            diagnostics[str(name)] = {
                "rhat": float(summary.loc[name, "R_hat"]),
                "ess_bulk": float(summary.loc[name, "ESS_bulk"]),
                "ess_tail": float(summary.loc[name, "ESS_tail"]),
            }
    if not diagnostics:
        raise GateError("the summary holds no global parameter")
    sampler = sampler or {}
    keep = trap["keep_index"]
    div = int(np.sum(sampler["divergent__"][:, keep])) if "divergent__" in sampler else 0
    hits = int(np.sum(sampler["treedepth__"][:, keep] >= max_treedepth)) if "treedepth__" in sampler else 0
    bfmi = e_bfmi(sampler["energy__"]) if "energy__" in sampler else []
    return GateReport(
        max_rhat=float(np.nanmax([d["rhat"] for d in diagnostics.values()])),
        min_ess_bulk=float(np.nanmin([d["ess_bulk"] for d in diagnostics.values()])),
        min_ess_tail=float(np.nanmin([d["ess_tail"] for d in diagnostics.values()])),
        chain_exclusion=trap,
        thresholds=th,
        divergences=div,
        max_treedepth_hits=hits,
        max_treedepth=max_treedepth,
        e_bfmi=bfmi,
        diagnostics=diagnostics,
        override=override.record() if override else None,
        n_draws_per_chain=int(lp_arr.shape[0]),
    )


@validation("VALIDATED")
def assess(
    csv_files: Sequence[str | Path], *, override: GateOverride | None = None, max_treedepth: int = 10
) -> GateReport:
    """Judge all chains of one fit together, as the study's `combine_fold` does.

    Reads lp__ from every chain, applies the chain-exclusion rule, runs `stansummary` on the
    retained chains and evaluates C2, C3 and the retained-chain count on the global parameters.

    Validation: VALIDATED.
    """
    files = chain_csvs_in_order(csv_files)
    sampler = read_sampler_columns(files)
    lp = sampler["lp__"]
    trap = retained_chains(lp, override=override) if override is None else None
    keep = trap["keep_index"] if trap is not None else retained_chains(lp, override=override)["keep_index"]
    if not keep:
        raise GateError("no chain was retained")
    summary = summary_on_chains(files, keep)
    if override is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GateOverrideWarning)
            return assess_tables(lp, summary, override=override, sampler=sampler, max_treedepth=max_treedepth)
    return assess_tables(lp, summary, override=None, sampler=sampler, max_treedepth=max_treedepth)
