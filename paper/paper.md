---
title: 'symphony-dt: governed Bayesian digital twins of disease progression in longitudinal clinical cohorts'
tags:
  - Python
  - Stan
  - Bayesian inference
  - digital twins
  - Parkinson's disease
  - disease progression modelling
  - uncertainty quantification
authors:
  - name: Ahmed Abdelmonem Hemedan
    orcid: 0000-0001-7403-181X
    affiliation: 1
affiliations:
  - name: Luxembourg Institute of Health, Strassen, Luxembourg
    index: 1
    ror: 012m8gv78
date: 25 November 2026
bibliography: paper.bib
---

# Summary

`symphony-dt` is a Python package for building, checking, calibrating, forecasting with and auditing Bayesian
digital twins of longitudinal clinical cohorts. A digital twin here is a fitted prediction model of one person's
disease course. It learns from a cohort, updates on the person's own clinic visits and returns forecasts with
uncertainty.

The package ships `pd_v2`, the population model of the SYMPHONY v2 study of Parkinson's disease progression. Its
fitted parameters describe how three clinical scores change over time. Motor signs are scored on Part III of the
Movement Disorder Society Unified Parkinson's Disease Rating Scale (MDS-UPDRS) [@goetz2008mdsupdrs]. Cognition is
scored on the Montreal Cognitive Assessment (MoCA) [@nasreddine2005moca]. Autonomic symptoms are scored on the
Scales for Outcomes in Parkinson's Disease, Autonomic (SCOPA-AUT) [@visser2004scopaaut]. The model is written in
Stan [@carpenter2017stan] and driven from Python.

The package also provides the steps a study needs to reach a defensible forecast. It harmonises data, applies fixed
convergence gates and scores held-out forecasts against simple baselines. It also checks the fitting procedure on
simulated data and records the provenance of every run. Each participant can also have an experimental twin. That
twin updates with each visit and checks whether its predicted probabilities stay calibrated. Every forecast is
associational, so the package makes no causal or treatment claim.

# Statement of need

Digital twins are proposed widely in medicine, yet few are evaluated with the rigour that clinical forecasting
needs [@laubenbacher2024digital; @niarakis2024immune]. A forecast is only useful if its stated uncertainty holds.
A 95% interval should contain the later score in about 95 of 100 cases.

Checking that property in practice takes much more than a fitted model. Clinic visits are irregular and often
incomplete, and scores pile up at the limits of their scales. Markov chain Monte Carlo can fail quietly, and
scoring rules chosen after the results are seen can flatter any method. When these steps lack fixed rules, two
analyses of one cohort can disagree for reasons unrelated to the disease.

`symphony-dt` serves disease-progression modellers, cohort data teams and methods researchers. It gives them one
place where these rules are fixed in code and tested, then recorded with every run. A team can therefore repeat a
study, or score a new method against it, on equal terms.

# State of the field

Open tools already address parts of this problem. Leaspy implements disease course mapping, a mixed-effects model
fitted along a disease timeline [@koval2021adcoursemap]. JMbayes fits Bayesian joint models of longitudinal and
time-to-event data [@rizopoulos2016jmbayes]. Stan and ArviZ supply general inference and diagnostics for Bayesian
models [@carpenter2017stan; @kumar2019arviz]. The simulation-based calibration rank plot checks a fitting
procedure on simulated data with known parameters [@talts2018sbc; @modrak2025sbc].

These tools are general, and that is their strength. They leave each study to choose its own convergence rules and
its own scoring. `symphony-dt` takes the opposite position for one model class. It freezes a model of record
together with the study's decision rules, and every result can then be audited against them.

The living twin builds on established methods. It updates a person's latent path with a sequential Monte Carlo
sampler [@delmoral2006smc]. An e-process tests whether its predicted probabilities stay calibrated
[@vovk2021evalues; @arnold2023sequential]. Adaptive conformal inference then recalibrates its intervals
[@gibbs2021adaptive]. Candidate visit times are ranked by expected information [@lindley1956information]. To our
knowledge, no open package joins these parts into one governed loop for a disease-progression twin. A full
literature review must still confirm that statement.

# Software design

`pd_v2` is a hierarchical state-space model with fitted population and participant parameters. Each participant
has a hidden severity in three domains, which can only rise. Its speed depends on a participant rate and on
baseline covariates. A sparse coupling matrix lets the current severity of one domain speed up another. Each score
is a noisy reading of one domain, bounded to the range of its instrument. Floor and ceiling values can be treated
as censored rather than truncated.

The package follows one design rule: anything that can change a study's result is fixed in code and checked.
Four mechanisms carry that rule.

**A frozen model of record.** The Stan programs are checked by their sha256 digest at import. New models are
registered beside `pd_v2` under their own names and cannot replace it.

**Fixed decision rules.** The convergence gates, a pre-declared chain-exclusion rule and the scoring specification
live in one place. Changing a gate needs a recorded override with a reason. Scoring refuses to run on a
specification that was not frozen first.

**A code of record.** The scoring code reproduces the study's scripts, and golden tests compare both on the same
synthetic folds. Interval coverage and the continuous ranked probability score [@gneiting2007scoring] are reported
against last observation carried forward and a linear mixed model.

**Evidence status and provenance.** Every public function carries one of three marks: validated, tested or
experimental. The mark states whether the code was used in the study on a clinical cohort, checked on synthetic
data only, or is still settling. Runs write manifests, data fingerprints and provenance records. A helper prompts reporting under the
2024 guideline for clinical prediction models that use statistical or machine learning methods [@collins2024tripodai].

Data enter through one validated long-format dataset. Adapters read wide tables and the Observational Medical
Outcomes Partnership (OMOP) common data model [@hripcsak2015ohdsi]. Further adapters cover Fast Healthcare
Interoperability Resources (FHIR), the Study Data Tabulation Model (SDTM) and the Medical Event Data Standard
(MEDS). Another reader handles the download layout of the Parkinson's Progression Markers Initiative (PPMI).

A single study file then runs every stage, from eligibility and an attrition table to a final report. Completed
stages are skipped, so a long run resumes where it stopped. Slurm job arrays, container images and portable run
bundles let one study run on a laptop or a cluster.

Governance is part of the design rather than a later step. No participant data ship with the package, and a privacy
gate checks every commit for them. Every test and tutorial uses the package's own simulator. Continuous
integration runs the test suite on Linux and macOS with Python 3.11 to 3.13. It also builds the documentation in
strict mode.

# Research impact statement

The first version of this framework was evaluated on the PPMI cohort [@marek2011ppmi]. That evaluation used
five-fold cross-validation and uncertainty-gated reporting. A preprint reports it in full [@hemedan2026clinic].
`pd_v2` develops that fitted prediction model further. It adds censoring at the MoCA ceiling and a levodopa dose
term in the motor reading. Version 2 of the package then wraps it in the governance layer described above.

The package is the analysis platform of the SYMPHONY v2 study, whose validation manuscript is in preparation.
External validation is planned in the Luxembourg Parkinson's Study (LuxPARK) of the National Centre of Excellence
in Research on Parkinson's Disease (NCER-PD). It will run under that cohort's data governance. Meanwhile, an open
synthetic benchmark compares the twin with simple baselines on cohorts with known truth.

# AI usage disclosure

To be completed by the author before submission.

# Acknowledgements

To be completed by the author before submission, with funding sources. The prediction model was developed on PPMI
data under the PPMI data use agreement.

# References
