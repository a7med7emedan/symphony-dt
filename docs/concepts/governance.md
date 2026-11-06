# Governance

A forecast is only as trustworthy as the record of how it was made. symphony-dt writes that record for you.

- **Run manifests** record the package version, the model digest and a fingerprint of the data. They also record
  the environment, the seeds, the CmdStan version, the gate verdicts and the times. They never hold data.
- **Data fingerprints** are sha256 digests of a canonical serialisation, so two runs can be shown to use the same
  data without sharing it.
- **Pre-registration.** A scoring specification is frozen by its hash before any target is scored. Scoring refuses
  an unfrozen or changed specification unless a deviation record names the change, the reason and the author.
- **Provenance** documents follow W3C PROV-JSON, linking inputs, activities, software and outputs by digest.
- **Reproduction.** `symphony-dt reproduce` reruns a recorded command and compares output digests.
- **Cards.** Model cards and data cards summarise what a model assumes and what a dataset holds, with small cells
  suppressed.
