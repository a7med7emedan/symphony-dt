# Data

## The canonical dataset

`TwinDataset` holds a cohort in long format. Time is whole days since each participant's baseline visit.

| frame | required columns | optional columns |
|---|---|---|
| `participants` | `participant_id`, `cohort` | `age_baseline_years`, `sex` (0 female, 1 male), `genotype_assay_available`, covariates |
| `observations` | `participant_id`, `time_days`, `endpoint`, `value` | `instrument_version` |
| `visits` | `participant_id`, `time_days` | `ledd` and other exposures |
| `markers` | `participant_id`, `name`, `value` | pathway or biomarker scores |

```python
from symphony_dt.data import TwinDataset
ds = TwinDataset(participants, observations, visits=visits)
ds.summary(min_cell=10)     # aggregate counts, small cells suppressed
ds.fingerprint()            # sha256 of the canonical dataset
```

Validation raises `DataValidationError` with a rule code. `DV001` is a missing column. `DV004` is a time that is
not a whole, non-negative number of days. `DV007` is a score outside its instrument range. `DV011` is a participant
without a baseline visit. `symphony_dt.data.VALIDATION_RULES` lists every code.

## Instruments

The `InstrumentRegistry` records each instrument's range, version, direction of worsening and boundary handling.
`default_registry()` holds the three pd_v2 instruments. Register new instruments for other models.

## Validity envelope

`check_envelope(dataset)` compares a dataset with the domain pd_v2 was designed for. It returns warnings with codes,
for example `ENV003` for an unknown cohort label or `ENV005` for an interval longer than three years. The limits
are design choices, not estimates from any cohort. A warning does not stop a fit; it says the model is used
outside the conditions its validation covers.

## Harmonisation

`harmonise(dataset, crosswalks)` converts instrument versions with user-supplied crosswalks. A `Crosswalk` must
carry the citation of its published source. No crosswalk ships with the package. `convert_units` rescales exposure
columns and records the conversion.

## The Stan data block

`build_stan_data(dataset, standardisation=..., pathway=...)` returns the pd_v2 data block. Fit the
`Standardisation` on training participants only, so no held-out value enters a centring constant. The builder
reproduces the study's builder exactly; a golden test checks this.

## Simulation

`simulate_cohort(n, seed=...)` draws parameters from the prior and simulates a cohort with the pd_v2 generative
model. It returns the dataset, the generating data block, the true parameters and the latent truths. The
underlying `simulate_replicate` reproduces the study's simulator draw for draw.
