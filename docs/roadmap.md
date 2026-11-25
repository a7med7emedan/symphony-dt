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

## What the package still lacks

This section audits pd_v2 and the package against what a leading Bayesian digital twin of a clinical cohort would
need. Items are grouped by theme and ordered within each theme by what they would add to the evidence. Many build
on methods that other tools already offer; the aim is to join them under the same gates and governance.

### 1. Evidence on people, not only on simulations

Every result so far is synthetic. The benchmark uses an oracle population, so it tests the participant machinery
but not population estimation.

- Publish the PPMI validation, then validate externally on NCER-PD/LuxPARK with covariate-shift diagnostics.
- Recalibrate for an external cohort and report coverage before and after.
- Run a full-size simulation-based calibration of pd_v2 at study settings, with its failure share reported.
- Report calibration by subgroup: sex, age band, cohort, genotype and site. (DONE: `subgroup_calibration`.)
- Score events with decision curve analysis as well as Brier score and AUC.
- Extend the benchmark to estimated populations, and compare with established methods. Candidates are disease
  course mapping in Leaspy, Bayesian joint models such as JMbayes2, Gaussian process models and neural sequence
  models.

### 2. Measurement that matches the instruments

pd_v2 reads each score as a bounded Gaussian. MDS-UPDRS Part III, MoCA and SCOPA-AUT are sums of ordinal items.

- An item-level graded response model (Samejima, 1969) for the instruments' items. It handles floors and ceilings
  by construction and copes with partly completed forms.
- A discrete likelihood on the sum score, beta-binomial or ordinal, as a lighter step.
- Examiner and site effects, and the recorded ON or OFF medication state at the motor examination.
- A transient visit-level term for practice effects. A monotone latent state cannot absorb a MoCA score that
  rises with repeated testing.

### 3. Biology the model does not yet see

- Biomarkers as further readings of the latent state. Candidates are DaTscan binding ratios, alpha-synuclein seed
  amplification, neurofilament light, MRI, smell tests and REM sleep behaviour disorder. These matter most in the
  prodromal stage, where clinical scores move least.
- A disease-time axis, as in disease course mapping (Schiratti and colleagues, 2017). Prodromal and PD
  participants could then share one timeline instead of one link offset.
- Heterogeneity beyond Gaussian rates: mixtures or subtype-specific coupling, and genetic groups such as GBA1 and
  LRRK2 carriers as hierarchical strata.
- Levodopa as pharmacology rather than a dose term. Holford and colleagues separate short and long duration
  responses, which a twin needs before it can describe medicated scores well.
- Frequent digital measures from wearables and smartphones, with their own measurement models.

### 4. People who leave the study

- A joint model in which the latent state drives the hazard of dropout and death. Informative loss to follow-up
  then enters the likelihood.
- Clinical milestones as time-to-event submodels: falls, dementia and loss of independence.
- Sensitivity analyses for data missing not at random, such as delta adjustment.

### 5. Inference at cohort scale

Every sub-step carries three innovation parameters, so the sampler's dimension grows with follow-up. The model card
also records slow mixing of the motor link slope and occasional trapped chains.

- Marginalise the innovations, by Kalman-type or Laplace approximations, or move to a continuous-time process with
  tractable increments.
- Reparametrise the motor link and initialise chains with Pathfinder, so fewer chains need excluding.
- Learn the population online, for example by SMC squared (Chopin, Jacob and Papaspiliopoulos, 2013). The twin
  registry could then relearn without a full refit.
- Scale within chains on many cores or on GPUs.

### 6. Model criticism and comparison

- Leave-future-out cross-validation (Bürkner, Gabry and Vehtari, 2020), the right criterion for forecasts.
- PSIS-LOO on held-out log predictive density, and stacking of the model arms (Yao and colleagues, 2018).
- Prior sensitivity by power scaling (Kallioinen and colleagues, 2024). The process-noise prior conflict on the
  model card is the first case to test.
- A per-parameter identifiability report from prior and posterior contraction.

### 7. From one Parkinson's model to a framework

- A declarative model builder. A specification file would name domains, instruments, links and coupling, and the
  package would generate, digest and register the Stan program. Other progressive conditions could then use the
  same gates.
- An R interface, export to ArviZ InferenceData and Leaspy's data format.
- Federated fitting across sites that cannot share data, for example by expectation propagation over partitions
  (Vehtari and colleagues, 2020).

### 8. Uses beyond forecasting, framed with care

- Prognostic scores for covariate adjustment in randomised trials, in the style of the PROCOVA method that the
  EMA qualified. Randomisation carries the causal weight; the twin supplies an associational prognosis.
- Visit scheduling by value of information, with visit cost and participant burden, across all endpoints.
- Any counterfactual use needs its own causal layer, with target trial emulation, stated assumptions and
  sensitivity analysis. That layer is out of scope for now.

### 9. Credibility for regulated settings

- A credibility plan in the style of ASME V&V 40, which the FDA's guidance on computational models also uses. It
  would state the context of use, the model risk and the evidence each claim needs.
- Risk of bias appraisal with PROBAST beside the existing TRIPOD+AI helper.
- An interactive participant report, tested with clinicians for how well it conveys uncertainty.
- A software life cycle suited to medical devices, only if a clinical use is ever pursued.

### Where to start

Three steps would add the most evidence per unit of work:

1. External validation with leave-future-out scoring and subgroup calibration.
2. An item-level measurement model with a joint dropout model.
3. Marginalised innovations, which would make the first two affordable at cohort scale.
