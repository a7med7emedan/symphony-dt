# Models

`get_model("pd_v2")` returns the registered model after checking the sha256 of its Stan programs.

| program | role | label |
|---|---|---|
| `model.stan` | the population model of record, byte-identical to the study | VALIDATED |
| `forecast_gq.stan` | standalone generated quantities: `y_rep_c`, `y_rep_p`, `mu_c` | TESTED |
| `conditional.stan` | one participant conditioned on fixed global parameters | EXPERIMENTAL use |

`ModelConfig.primary()` sets the study's primary arm: A01 off, A02 on, A03 on for MoCA only.
`ModelConfig.a01_sensitivity()` sets the pre-declared sensitivity arm.

New models register beside pd_v2 with `register_model(ModelSpec(...))`, each with its own digests. pd_v2 can never
be replaced. Read its assumptions and known limits on the [model card](../model-card.md).
