"""A TRIPOD+AI reporting helper for users who report their own prediction studies.

The headings paraphrase the sections of TRIPOD+AI (Collins GS et al. BMJ 2024;385:e078378).
They are prompts, not the statement itself. Use the published statement for the authoritative
wording and item numbering.
"""

from __future__ import annotations

from collections.abc import Mapping

from symphony_dt.labels import validation

__all__ = ["TRIPOD_AI_CITATION", "tripod_ai_items", "tripod_ai_report"]

TRIPOD_AI_CITATION = (
    "Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting "
    "clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378."
)

_ITEMS: tuple[tuple[str, str, str], ...] = (
    (
        "title",
        "Title",
        "Identify the study as developing or evaluating a prediction model, the population and the outcome.",
    ),
    ("abstract", "Abstract", "Give a structured summary following the TRIPOD+AI abstract guidance."),
    ("background", "Background", "Explain the clinical context and why a model is needed."),
    ("objectives", "Objectives", "State whether the study develops, evaluates, or both."),
    (
        "data_sources",
        "Data sources",
        "Describe each data source, its dates and its setting, for example PPMI development "
        "and NCER-PD/LuxPARK external.",
    ),
    ("participants", "Participants", "Give eligibility criteria and how participants were selected."),
    ("data_preparation", "Data preparation", "Describe preprocessing, harmonisation and quality checks."),
    ("outcome", "Outcome", "Define each outcome, how and when it was measured, and any blinding."),
    ("predictors", "Predictors", "Define each predictor, how and when it was measured."),
    ("sample_size", "Sample size", "Explain how the sample size was arrived at."),
    ("missing_data", "Missing data", "Describe how missing data were handled."),
    (
        "analysis",
        "Analytical methods",
        "Describe the model, its fitting, its convergence checks and its evaluation.",
    ),
    (
        "performance",
        "Performance measures",
        "Name calibration and discrimination measures and how uncertainty was quantified.",
    ),
    ("fairness", "Fairness", "Describe how performance was examined across subgroups."),
    (
        "model_output",
        "Model output",
        "Describe what the model outputs, for example predictive distributions.",
    ),
    (
        "development_vs_evaluation",
        "Development versus evaluation",
        "State differences between development and evaluation data.",
    ),
    ("ethics", "Ethical approval", "Name the approving body and data use agreements."),
    ("funding", "Funding", "Give the sources of funding and their role."),
    ("conflicts", "Conflicts of interest", "Declare conflicts of interest."),
    ("protocol", "Protocol and registration", "Say where the protocol or pre-registration can be found."),
    ("data_sharing", "Data sharing", "Say how data can be accessed, or why they cannot be shared."),
    ("code_sharing", "Code sharing", "Say where the code is, with its version and digest."),
    ("ppi", "Patient and public involvement", "Describe any patient and public involvement."),
    (
        "results_participants",
        "Participants in results",
        "Report flow, characteristics and follow-up per data source.",
    ),
    ("model_specification", "Model specification", "Give the full model so others can use it."),
    (
        "model_performance",
        "Model performance",
        "Report performance with uncertainty, overall and by subgroup.",
    ),
    ("interpretation", "Interpretation", "Interpret results against objectives and prior evidence."),
    ("limitations", "Limitations", "Discuss limitations, including bias and generalisability."),
    ("usability", "Usability", "Discuss intended use and the conditions under which it does not apply."),
)


@validation("TESTED")
def tripod_ai_items() -> list[dict[str, str]]:
    """Return the checklist prompts as dicts with `key`, `heading` and `prompt`.

    Validation: TESTED.
    """
    return [{"key": k, "heading": h, "prompt": p} for k, h, p in _ITEMS]


@validation("TESTED")
def tripod_ai_report(answers: Mapping[str, str], *, title: str = "TRIPOD+AI checklist") -> str:
    """Return a Markdown checklist with the user's answers; unanswered prompts are marked TO DO.

    Unknown keys in `answers` raise KeyError, so a typo cannot hide an item.

    Validation: TESTED.
    """
    keys = {k for k, _, _ in _ITEMS}
    unknown = sorted(set(answers) - keys)
    if unknown:
        raise KeyError(f"unknown checklist keys {unknown}")
    lines = [
        f"# {title}",
        "",
        f"Source: {TRIPOD_AI_CITATION}",
        "",
        "| heading | prompt | where reported |",
        "|---|---|---|",
    ]
    for k, h, p in _ITEMS:
        lines.append(f"| {h} | {p} | {answers.get(k, 'TO DO')} |")
    done = sum(1 for k in keys if answers.get(k))
    lines += ["", f"{done} of {len(keys)} prompts answered."]
    return "\n".join(lines) + "\n"
