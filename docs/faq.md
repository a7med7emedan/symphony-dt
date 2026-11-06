# FAQ

**Does the package include PPMI data?**
No. It contains no participant data of any kind. See [Governance and data access](governance.md).

**Can I change the pd_v2 Stan program?**
No. It is the frozen model of record, checked by its sha256. Register a new model under a new name instead.

**My fit does not converge. Can I relax the gates?**
Run longer chains first. If you must change a threshold, use a `GateOverride` with a reason and an author. The
change is written into the report and the manifest.

**Why does scoring ask for a frozen specification?**
Fixing the scoring rules before seeing the scores prevents choosing rules that flatter the result. A change after
freezing needs a deviation record.

**Which forecast route should I use?**
Use the exact route for evaluation and reporting. The fast route is EXPERIMENTAL and suits exploration.

**Is the twin a clinical tool?**
No. It produces associational forecasts for research. It makes no causal or treatment claim.

**How do I cite it?**
See [Citation](citation.md).
