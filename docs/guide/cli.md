# Command line

| command | purpose |
|---|---|
| `symphony-dt about` | logo, version, author, citation and documentation address |
| `symphony-dt check [--compile]` | frozen model digests, CmdStan, optional compilation |
| `symphony-dt validate-data PATH` | validate a dataset folder or a wide table |
| `symphony-dt envelope PATH` | check a dataset against the pd_v2 design envelope |
| `symphony-dt simulate --n N --out DIR` | simulate a synthetic cohort, with a manifest |
| `symphony-dt split DATA --out DIR` | hold out participants after an anchor |
| `symphony-dt fit DATA --out DIR` | fit pd_v2 and print the gate verdict |
| `symphony-dt gates FIT` | evaluate the gates on a fit folder |
| `symphony-dt forecast FIT SPLIT --out DIR` | exact-route forecasts of the held-out targets |
| `symphony-dt evaluate SPLIT GQ_CSV --out DIR` | score against LOCF and the LMM under a frozen specification |
| `symphony-dt sbc` | a small prior-regime SBC campaign for pd_v2 |
| `symphony-dt ppc FIT` | posterior predictive check of a fit on its own data |
| `symphony-dt report MANIFEST...` | an HTML report from run manifests |
| `symphony-dt reproduce MANIFEST` | rerun a recorded command and compare digests |

Commands print aggregates only.
