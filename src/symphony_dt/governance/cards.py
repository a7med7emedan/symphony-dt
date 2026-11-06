"""Model card and data card generators.

The data card reports aggregate counts only and suppresses any cell below a minimum number of
participants, so it can be shared where the data cannot.
"""

from __future__ import annotations

from typing import Any

from symphony_dt.data.dataset import TwinDataset
from symphony_dt.data.envelope import check_envelope
from symphony_dt.labels import validation

__all__ = ["data_card_markdown", "model_card_markdown"]


@validation("TESTED")
def model_card_markdown(name: str = "pd_v2") -> str:
    """Return the model card of a registered model, with its verified digests appended.

    Validation: TESTED.
    """
    from symphony_dt.models import get_model, model_card

    spec = get_model(name)
    lines = [model_card(name).rstrip(), "", "## Verified digests", ""]
    lines += [f"- `{role}`: `{digest}`" for role, digest in sorted(spec.digests.items())]
    return "\n".join(lines) + "\n"


@validation("TESTED")
def data_card_markdown(
    dataset: TwinDataset, *, title: str = "Data card", min_cell: int = 10, source: str = ""
) -> str:
    """Return a data card: aggregate counts, envelope warnings and the dataset fingerprint.

    Any count of participants below `min_cell` is shown as `<min_cell`.

    Validation: TESTED.
    """
    summary: dict[str, Any] = dataset.summary(min_cell=min_cell)
    env = check_envelope(dataset)
    lines = [f"# {title}", ""]
    if source:
        lines += [f"**Source.** {source}", ""]
    lines += [
        f"**Fingerprint.** `{dataset.fingerprint()}` (sha256 of the canonical dataset; the data are "
        "not included).",
        "",
        "## Counts",
        "",
        "| quantity | value |",
        "|---|---|",
    ]
    for key in (
        "participants",
        "observations",
        "visits",
        "median_visits_per_participant",
        "max_follow_up_years",
    ):
        lines.append(f"| {key.replace('_', ' ')} | {summary[key]} |")
    lines += ["", "## Cohorts", "", "| cohort | participants |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["cohorts"].items()]
    lines += ["", "## Observations by endpoint", "", "| endpoint | observations |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["observations_by_endpoint"].items()]
    lines += [
        "",
        "## Instruments",
        "",
        "| name | version | range | worsening | boundary |",
        "|---|---|---|---|---|",
    ]
    for inst in dataset.instruments:
        lines.append(
            f"| {inst.name} | {inst.version} | {inst.lower:g} to {inst.upper:g} | "
            f"{inst.worsening} | {inst.boundary} |"
        )
    lines += ["", "## Validity envelope (pd_v2)", ""]
    if env.inside:
        lines.append("No out-of-envelope feature was found.")
    else:
        lines += [f"- `{w.code}` {w.message}" for w in env.warnings]
    return "\n".join(lines) + "\n"
