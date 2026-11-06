# Input and output

`read_table` and `write_table` handle CSV, TSV, Parquet, Excel and Arrow, chosen by file suffix. Both accept
pandas or polars frames. `write_dataset` stores a `TwinDataset` as one table per frame plus `dataset.json`, and
`read_dataset` checks the fingerprint on the way back.

```python
from symphony_dt.io import write_dataset, read_dataset
write_dataset(ds, "cohort", format="parquet")
same = read_dataset("cohort")
```

Posteriors are stored as netCDF through ArviZ: `write_posterior(idata, "fit.nc")` and `read_posterior("fit.nc")`.

Adapters for wide tables, OMOP CDM, FHIR R4, CDISC SDTM and the PPMI download layout are described on the
[Interoperability](../interoperability.md) page.
