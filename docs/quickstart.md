# Quickstart

This example simulates a synthetic cohort, holds out late visits, fits pd_v2 and scores the forecasts.
It uses no real data. Short chains keep it fast; they will not pass the convergence gates, and the gates say so.

```python
import numpy as np
from symphony_dt.data import simulate_cohort
from symphony_dt.forecast import split_at_anchor, forecast_exact
from symphony_dt.fit import fit
from symphony_dt.evaluate import ScoringSpec, freeze_spec, score_fold

cohort = simulate_cohort(60, seed=1)
tests = sorted(np.random.default_rng(0).choice(60, 15, replace=False).tolist())
split = split_at_anchor(cohort.stan_data, tests, anchor_years=2.0)
result = fit(split.fit, chains=4, iter_warmup=300, iter_sampling=300, output_dir="run")
print("\n".join(result.gates().summary_lines()))
forecast = forecast_exact(result, split, total_keep=400)
spec = ScoringSpec.default()
scores = score_fold(split.fit.data, split.targets.data, split.meta, forecast.draws, tests,
                    spec=spec, frozen=freeze_spec(spec))
print("\n".join(scores.lines[:3]))
```

The same steps run from the command line:

```bash
symphony-dt simulate --n 60 --out sim
symphony-dt split sim/stan_data.json --out split
symphony-dt fit split/stan_data_fit.json --out fit --warmup 300 --draws 300
symphony-dt forecast fit split --out forecast
symphony-dt evaluate split forecast/gq.csv --out scores
```

For a study-quality fit, use the study's settings: at least four chains, 1,000 warm-up and 1,000 sampling
iterations. Report the gate verdict whatever it is.
