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

**Can I model a disease other than Parkinson's disease?**
Not with pd_v2, whose three domains and instruments are fixed. Write a new Stan program and register it beside
pd_v2 with `register_model`, under its own name and digests. See [Models](guide/models.md).

**Can I use my own cohort?**
Yes. Map it with a shipped mapping or your own, then check it with `symphony-dt envelope`. A cohort outside the
design envelope may still fit, but pd_v2 was not developed for it. See [Interoperability](interoperability.md).

**How long does a fit take?**
Cost grows with participants, visits and sub-steps, because every sub-step adds three latent innovations to the
sampler. Small synthetic cohorts suit a laptop. Full cohorts at study settings suit a cluster; see
[Running on any machine](guide/compute.md).

**Is the twin a clinical tool?**
No. It produces associational forecasts for research. It makes no causal or treatment claim.

**How do I cite it?**
See [Citation](citation.md).
