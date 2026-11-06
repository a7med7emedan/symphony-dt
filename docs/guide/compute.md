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

**Containers** install symphony-dt and CmdStan 2.39.0 and compile pd_v2 at build time.

**The twin service** reads commands such as `{"op": "observe", "visit": {...}}`, `{"op": "forecast",
"horizons": [1, 2]}` and `{"op": "next_visit"}`. It answers each with one JSON line and saves the twin after every
observation.
