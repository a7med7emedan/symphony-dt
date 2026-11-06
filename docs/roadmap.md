# Roadmap

This page plans what a computational modeller needs from symphony-dt beyond 2.0. It records what comparable
tools already do, what we borrow, and what each release delivers. Items marked DONE ship in the release named.

## What comparable tools offer

| tool | what it does well | what we borrow |
|---|---|---|
| Leaspy (disease progression modelling) | long-format data container; population and individual trajectory plots | a cohort spaghetti plot and a per-participant report figure |
| OHDSI CohortGenerator | inclusion rules applied in sequence; an attrition table with a count after each rule and the gain of each rule | sequential eligibility rules with the same attrition table and per-rule gain |
| MEDS (Medical Event Data Standard) | one flat event table: subject, time, code, numeric value | a MEDS reader and writer |
| bayesplot and ArviZ | trace, rank, energy and predictive interval plots | the same diagnostic plots on our gated chains |
| SBC (Modrak et al.) | rank plots with simultaneous bands | already present; extended with reliability plots for events |
| e-value calibration tests (Arnold, Henzi and Ziegel) | sequential, anytime-valid tests of forecast calibration | the monitor that decides when the twin relearns |
| adaptive conformal inference (Gibbs and Candes, 2021) | long-run coverage under drift | online recalibration of the twin's intervals |
| probabilistic digital twins in engineering | belief updating and information-driven actions | visit scheduling by expected information gain |

The survey drew on the [Leaspy documentation](https://leaspy.readthedocs.io/en/stable/auto_examples/plot_02_parkinson_example.html),
[CohortGenerator attrition](https://www.rdocumentation.org/packages/CohortGenerator/versions/1.1.0/topics/computeCohortAttrition),
the [MEDS package](https://pypi.org/project/meds) and the [bayesplot reference](https://mc-stan.org/bayesplot/reference/index.html).

## Release 2.1: the modeller's workflow

### 1. Eligibility with an attrition table (DONE in 2.1)

Declarative inclusion rules apply in order. They cover minimum visits, minimum follow-up and endpoints required at
baseline. They also cover cohort labels, age range, the longest gap between visits and custom rules.
The result holds the eligible dataset, a sequential attrition table with each rule's gain, and a flow chart.
Counts below the small-cell limit are suppressed in every shared output.

### 2. More data formats and a data profile (DONE in 2.1)

Readers for SAS, Stata, SPSS, JSON Lines and SQL queries, besides CSV, Parquet, Excel and Arrow. A MEDS reader
and writer. A REDCap mapping template. A data profile that reports coverage, visits, intervals, missingness
and boundary shares per endpoint, with a figure.

### 3. One study file, one command (DONE in 2.1)

A `study.yaml` controls the whole run. It names the data source and mapping, the eligibility rules and the model
switches. It also sets the held-out design, the sampler, any gate override, forecasts, scoring and outputs.
`symphony-dt init` writes a commented template. `symphony-dt run study.yaml` runs every stage, writes a manifest
per stage and skips stages that are already complete. `--dry-run` validates the file and the data without fitting.

### 4. Outputs a modeller can use (DONE in 2.1)

Tidy tables for the posterior summary, forecasts and scores, written as CSV, Parquet and one Excel workbook.

### 5. Plots for every step (DONE in 2.1)

Trace, rank and energy plots; posterior predictive intervals; and coverage against nominal level. Reliability of
event probabilities, cohort trajectories, the attrition flow, the data profile and a one-page participant report.
Every plot uses the colour-blind-safe palette and saves to PNG, PDF and SVG in one call.

### 6. A twin that learns and interacts (DONE in 2.1, EXPERIMENTAL)

A `ParticipantTwin` object holds one person's twin for its whole life:

- **observe** a new visit and update the person's latent state at once, without refitting the population. The
  update is a tempered SMC sampler with preconditioned Crank-Nicolson moves (Del Moral, Doucet and Jasra, 2006);
- **forecast** any horizon, with uncertainty split into population, person and measurement parts;
- **suggest the next visit** that most reduces forecast uncertainty, by expected information gain (Lindley, 1956;
  Chaloner and Verdinelli, 1995);
- **milestone curves**: the probability of crossing a clinical threshold by each time;
- **an audit log** of every update, saved and reloaded with the twin.

A `TwinMonitor` watches every forecast that later meets its outcome:

- an **anytime-valid calibration monitor**: an e-process on PIT values (Arnold, Henzi and Ziegel; Vovk and Wang).
  It may be checked after every visit, and its false-alarm rate stays at alpha for all time;
- **adaptive conformal recalibration** (Gibbs and Candes, 2021) that widens or narrows intervals online to keep
  long-run coverage on target, on the bounded clinical scale;
- a **relearning policy**: when the monitor's evidence passes its threshold, the population model is refitted,
  and the new posterior is versioned with its lineage.

**Why this is new.** Each component exists in the literature. To our knowledge, no tool joins them into one
governed loop for a disease-progression twin. The loop updates a monotone latent state, monitors calibration with
error control, recalibrates on bounded scales and schedules visits by information. A visit-triggered PD twin on
the same three scores appears in a 2026 preprint. The monitoring, recalibration and scheduling layers go beyond it. This claim needs a full
literature review before publication, and every part is labelled EXPERIMENTAL until validated on a cohort.

### 7. Runs on any machine (DONE in 2.1)

- One run bundle holds everything needed to repeat a run elsewhere: data block, study file, manifest, digests.
- Slurm job scripts with one chain per task, an Apptainer definition for HPC and a Dockerfile for any machine.
- JSON Schemas for the dataset and the study file, so R, Julia or any language can produce valid inputs.
- A JSON command interface for twins: any program can send a visit and receive a forecast on standard streams.

### 8. An open benchmark (DONE in 2.1)

Synthetic scenarios with known truth cover dense and sparse visits, floor effects, informative dropout and drift.
Each is scored with the same frozen specification for the twin, LOCF and the mixed model. One command runs it and writes a
table, so any method can be compared on equal terms.

## Later

- Covariate-shift diagnostics and recalibration for external cohorts such as NCER-PD/LuxPARK.
- Prognostic scores for covariate adjustment in trials (PROCOVA-style), framed as associational.
- A joint model of dropout, so informative loss to follow-up enters the likelihood.
- Interoperability with Leaspy's data format and an R interface.
- Model comparison by PSIS-LOO on held-out log predictive density.
- Prior sensitivity by power scaling.
- An interactive report that explores one participant's twin.
