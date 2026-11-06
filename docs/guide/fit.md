# Fitting and gates

```python
from symphony_dt.fit import fit
result = fit(stan_data, chains=4, iter_warmup=1000, iter_sampling=1000, seed=20260729,
             threads_per_chain=1, output_dir="fit")
report = result.gates()
print(report.converged)
```

Chains run in parallel and are written to `fit/chains/`. With `resume=True`, complete chains are kept and only the
rest run. They use the same seed and chain id, so a resumed fit equals an uninterrupted one. Every fit writes
`fit_manifest.json` with the model digest, the data digest, the seeds and the environment.

`inits="prior"` starts each chain at the prior location with a small per-chain jitter. `inits="random"` uses
CmdStan's default, which the study's held-out fits used.

`result.to_inference_data()` returns an ArviZ object holding the global parameters.

## Gates

`result.gates()` applies the [convergence gates](../concepts/gates.md) to every chain together. The report lists
the retained chains, max R-hat, min bulk and tail ESS, divergences, tree depth hits and E-BFMI. To change a
threshold, pass a `GateOverride` with a reason and an author; the change is logged.
