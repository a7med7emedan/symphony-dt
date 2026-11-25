# Command line

Every command prints aggregates only. Run `symphony-dt COMMAND --help` for its options.

## Set up and check

| command | purpose |
|---|---|
| `symphony-dt about` | logo, version, author, citation and documentation address |
| `symphony-dt check [--compile]` | frozen model digests, CmdStan, optional compilation |

## Whole studies

| command | purpose |
|---|---|
| `symphony-dt init DIR` | write a commented `study.yaml` template |
| `symphony-dt run STUDY [--dry-run] [--stages S] [--force]` | run a study file stage by stage, skipping complete stages |

See [the modeller's workflow](workflow.md) for the study file and its stages.

## Data

| command | purpose |
|---|---|
| `symphony-dt validate-data PATH` | validate a dataset folder or a wide table |
| `symphony-dt envelope PATH` | check a dataset against the pd_v2 design envelope |
| `symphony-dt eligibility PATH --rules RULES` | apply eligibility rules and print the attrition table |
| `symphony-dt profile PATH` | coverage, visits, intervals and boundary shares per endpoint |
| `symphony-dt simulate --n N --out DIR` | simulate a synthetic cohort, with a manifest |
| `symphony-dt schema NAME` | the JSON Schema of the study file, the dataset tables or a twin visit |

## Fit, forecast and score

| command | purpose |
|---|---|
| `symphony-dt split DATA --out DIR` | hold out participants after an anchor |
| `symphony-dt fit DATA --out DIR` | fit pd_v2 and print the gate verdict |
| `symphony-dt gates FIT` | evaluate the gates on a fit folder |
| `symphony-dt forecast FIT SPLIT --out DIR` | exact-route forecasts of the held-out targets |
| `symphony-dt evaluate SPLIT GQ_CSV --out DIR` | score against LOCF and the LMM under a frozen specification |

## Calibration and benchmark

| command | purpose |
|---|---|
| `symphony-dt sbc` | a small prior-regime SBC campaign for pd_v2 |
| `symphony-dt ppc FIT` | posterior predictive check of a fit on its own data |
| `symphony-dt bench --out FILE` | the open synthetic benchmark, written as a tidy table |

## Governance

| command | purpose |
|---|---|
| `symphony-dt report MANIFEST...` | an HTML report from run manifests |
| `symphony-dt reproduce MANIFEST` | rerun a recorded command and compare digests |

## Compute

| command | purpose |
|---|---|
| `symphony-dt slurm DATA --out DIR` | an sbatch job array with one chain per task, then the gates |
| `symphony-dt container KIND` | a Dockerfile (`docker`) or an Apptainer definition (`apptainer`) with CmdStan |
| `symphony-dt bundle write FILES... --out FILE` | pack files with SHA256SUMS and an environment lock |
| `symphony-dt bundle verify FILE` | check every file of a bundle against its digest |

See [running on any machine](compute.md) for worked examples.

## Living twins (EXPERIMENTAL)

| command | purpose |
|---|---|
| `symphony-dt twin create` | create a twin from a population fit and one participant's history |
| `symphony-dt twin forecast TWIN` | forecast quantiles and the uncertainty split |
| `symphony-dt twin next-visit TWIN` | rank candidate visit times by expected reduction in forecast variance |
| `symphony-dt twin serve TWIN` | serve a saved twin over JSON lines on standard input and output |

See [the living twin](twin.md) for what each step computes.
