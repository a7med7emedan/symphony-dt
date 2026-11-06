# Running on any machine

| need | command |
|---|---|
| repeat a run elsewhere | `symphony-dt bundle write files... --out run.tar.gz`, then `symphony-dt bundle verify run.tar.gz` |
| an HPC cluster with Slurm | `symphony-dt slurm stan_data.json --out fit --chains 8 > fit.sbatch`, then `sbatch fit.sbatch` |
| a container | `symphony-dt container docker > Dockerfile` or `symphony-dt container apptainer > symphony.def` |
| inputs from R, Julia or any language | `symphony-dt schema study`, `schema dataset` or `schema visit` |
| a twin driven by another program | `symphony-dt twin serve twin.npz`, one JSON command per line |

**Bundles** hold the files you name, with a SHA256SUMS file, the installed package versions and a README. Verify a
bundle on the receiving machine before you trust a rerun.

**Slurm.** The job array runs one CmdStan chain per task with the same seed and its own chain id. A resumed or
split run therefore equals a local one. The first task submits a dependent job that judges the gates once every
chain has finished. Chains never overwrite one another.

**Containers** install symphony-dt and CmdStan 2.39.0 and compile pd_v2 at build time. A ready image is published
for every release, for `linux/amd64` and `linux/arm64`, with a signed build provenance:

```bash
docker pull ghcr.io/a7med7emedan/symphony-dt:2.1.0
docker run --rm -v "$PWD":/work ghcr.io/a7med7emedan/symphony-dt:2.1.0 simulate --n 60 --out cohort
docker run --rm -v "$PWD":/work ghcr.io/a7med7emedan/symphony-dt:2.1.0 fit cohort/stan_data.json --out fit
apptainer pull symphony-dt.sif docker://ghcr.io/a7med7emedan/symphony-dt:2.1.0
gh attestation verify oci://ghcr.io/a7med7emedan/symphony-dt:2.1.0 --owner a7med7emedan
```

The image runs as an unprivileged user. On Linux, add `--user "$(id -u):$(id -g)"` so that output files belong to
you. The tags are the version (`2.1.0`), the minor series (`2.1`) and `latest`.

**The twin service** reads commands such as `{"op": "observe", "visit": {...}}`, `{"op": "forecast",
"horizons": [1, 2]}` and `{"op": "next_visit"}`. It answers each with one JSON line and saves the twin after every
observation.
