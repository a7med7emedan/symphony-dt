# %% [markdown]
# # 4. Governance and reports
#
# Every result should travel with a record of how it was made. This tutorial writes a run manifest, a provenance
# document, a frozen specification with a deviation record, a TRIPOD+AI checklist and an HTML report.

# %%
from pathlib import Path

from symphony_dt.data import simulate_cohort
from symphony_dt.governance import ProvenanceDocument, RunManifest

out = Path("tutorial_governance")
out.mkdir(exist_ok=True)
cohort = simulate_cohort(50, seed=9)
manifest = RunManifest.create("simulate", model="pd_v2", data_sha256=cohort.stan_data.sha256(),
                              dataset_fingerprint=cohort.dataset.fingerprint(), seeds={"simulate": 9})
manifest.write(out / "manifest.json")
{k: v for k, v in manifest.to_dict().items() if k in ("kind", "package_version", "dataset_fingerprint")}

# %%
doc = ProvenanceDocument.for_run("simulate", {}, {"stan_data.json": cohort.stan_data.sha256()})
doc.write(out / "provenance.json")
list(doc.to_dict())

# %% [markdown]
# ## Pre-registration

# %%
from symphony_dt.errors import UnfrozenSpecificationError
from symphony_dt.evaluate import DeviationRecord, ScoringSpec, freeze_spec, require_frozen

spec = ScoringSpec.default()
frozen = freeze_spec(spec, out / "frozen_spec.json")
changed = spec.model_copy(update={"min_bin": 20})
try:
    require_frozen(changed, frozen)
except UnfrozenSpecificationError as err:
    print("refused:", err)
require_frozen(changed, frozen, DeviationRecord("smaller cohort", "A. Analyst", {"min_bin": 20}))["deviation"]

# %% [markdown]
# ## TRIPOD+AI checklist

# %%
from symphony_dt.governance import tripod_ai_report

print(tripod_ai_report({"title": "Title page", "code_sharing": "Methods, code availability"})[:600])

# %% [markdown]
# ## HTML report

# %%
from symphony_dt.report import write_html_report

write_html_report(out / "report.html", [manifest.to_dict()], title="Synthetic run",
                  notes=["Synthetic data only."])
