# Governance and reports

```python
from symphony_dt.governance import RunManifest, ProvenanceDocument, data_card_markdown
manifest = RunManifest.create("evaluate", model="pd_v2", dataset_fingerprint=ds.fingerprint(),
                              seeds={"bootstrap": 20261021})
manifest.write("manifest.json")
ProvenanceDocument.for_run("evaluate", {"data": ds.fingerprint()}, {"scores": scores.sha256()}).write("prov.json")
print(data_card_markdown(ds, source="my cohort"))
```

- `freeze_spec`, `require_frozen` and `DeviationRecord` implement [pre-registration](../concepts/governance.md).
- `model_card_markdown` returns the model card with verified digests.
- `tripod_ai_items` and `tripod_ai_report` help you report your own study against TRIPOD+AI.
- `reproduce(manifest)` reruns a recorded CLI command and compares output digests.
- `symphony_dt.report.write_html_report` writes a self-contained HTML report.
