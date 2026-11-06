# Interoperability

Every adapter flattens its source into named tables and applies a YAML mapping. A mapping names tables,
identifier columns, the time column and its unit, and how codes map to endpoints. It never holds data. Copy a
shipped mapping, edit it and pass the file path. All examples below are synthetic.

```python
from symphony_dt.io import list_mappings, load_mapping
list_mappings()   # ['fhir', 'generic_long', 'omop', 'ppmi', 'sdtm', 'wide']
```

Time is re-expressed as whole days since each participant's first record. Rows with a missing value or time are
dropped and counted in `dataset.metadata["mapping_dropped"]`.

## Wide tables

One row per visit, one column per score.

```python
import pandas as pd
from symphony_dt.io import from_wide
frame = pd.DataFrame({"participant_id": ["p1", "p1"], "cohort": ["PD", "PD"], "age": [61, 61],
                      "sex": [0, 0], "days": [0, 365], "updrs3": [20, 24], "moca": [27, 26],
                      "scopa_aut": [8, 9], "ledd": [300, 400]})
ds = from_wide(frame)
```

## A generic long table, for any cohort

The `generic_long` mapping reads a participants table and a long scores table, for example from a national cohort
such as NCER-PD/LuxPARK. Time may be in days, weeks, months or years.

```python
from symphony_dt.io import from_mapping
ds = from_mapping({"participants": participants, "scores": scores}, "my_cohort.yaml")
```

## PPMI study data

You obtain PPMI data yourself; see [Governance and data access](governance.md). `from_ppmi_export(folder)` reads
the curated exports the `ppmi` mapping names: participant status, demographics, MDS-UPDRS Part III, MoCA and
SCOPA-AUT. Identifiers are taken from the PATNO column and dates from INFODT. Two columns must be prepared by you:
`AGE_AT_BL` and `SCAU_TOTAL`, the SCOPA-AUT total on the 0 to 69 scale. No proration or derived value ships.

## OMOP CDM

`from_omop({"person": ..., "measurement": ...}, mapping)` reads PERSON and MEASUREMENT. The shipped mapping has no
concept identifiers; set the concept ids your vocabulary uses for each total score. Age is taken at the first
record from `year_of_birth`; sex from `gender_concept_id` (8532 female, 8507 male).

## HL7 FHIR R4

`from_fhir(bundles, mapping)` reads Patient and Observation resources from Bundles given as dicts, JSON strings or
paths. Codes are matched as `system#code`; the shipped codes are placeholders to replace with the codes your
terminology service confirms.

## CDISC SDTM

`from_sdtm(qs, dm, mapping)` reads the QS and DM domains. Time is the study day `QSDY`. Test codes are matched on
`QSTESTCD`; the shipped values are examples to adapt to your controlled terminology.
