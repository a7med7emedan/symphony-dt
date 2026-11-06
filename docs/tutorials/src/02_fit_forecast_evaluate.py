# %% [markdown]
# # 2. Fit, check the gates, forecast and evaluate
#
# We hold out each test participant's visits after two years, fit pd_v2 on the rest, forecast the held-out visits
# and score the forecasts against LOCF and a linear mixed model. The chains are short so the tutorial runs in a
# few minutes. Short chains usually fail the gates, and the report says so.

# %%
import numpy as np

from symphony_dt.data import simulate_cohort
from symphony_dt.forecast import split_at_anchor

cohort = simulate_cohort(80, seed=3)
ids = cohort.dataset.participant_ids
part = cohort.dataset.participants.set_index("participant_id").loc[ids]
tests = sorted(np.random.default_rng(0).choice(80, 20, replace=False).tolist())
split = split_at_anchor(cohort.stan_data, tests, anchor_years=2.0,
                        cohort=part["cohort"].tolist(), sex=part["sex"].tolist())
print(len(split.meta), "held-out targets from", len(tests), "test participants")

# %% [markdown]
# ## Fit pd_v2

# %%
from symphony_dt.fit import fit

result = fit(split.fit, chains=4, iter_warmup=300, iter_sampling=300, seed=11, output_dir="tutorial_fit")
report = result.gates()
print("\n".join(report.summary_lines()))

# %%
report.lowest_ess(5)

# %% [markdown]
# ## Posterior of the population rates

# %%
import pandas as pd

from symphony_dt.plots import plot_effects

frame = pd.DataFrame({name: result.draws(name).ravel() for name in
                      ["mu_r.1", "mu_r.2", "mu_r.3", "c_MM", "c_CC", "c_NN"]})
plot_effects(frame);

# %% [markdown]
# ## Exact-route forecasts

# %%
from symphony_dt.forecast import forecast_exact

forecast = forecast_exact(result, split, total_keep=400, output_dir="tutorial_forecast")
forecast.quantiles().head()

# %%
forecast.plot(participant=int(split.meta["i"].iloc[0]));

# %% [markdown]
# ## Score against LOCF and the linear mixed model
#
# The specification is frozen before scoring. Scoring refuses an unfrozen specification.

# %%
from symphony_dt.evaluate import ScoringSpec, freeze_spec, score_fold

spec = ScoringSpec.default()
frozen = freeze_spec(spec)
scores = score_fold(split.fit.data, split.targets.data, split.meta, forecast.draws, tests,
                    spec=spec, frozen=frozen)
print("\n".join(scores.lines[:4]))

# %%
from symphony_dt.plots import plot_coverage_by_window, plot_crps_differences

plot_coverage_by_window(scores.scores);

# %%
plot_crps_differences(scores.scores);

# %% [markdown]
# ## The fast route (EXPERIMENTAL)
#
# Forecast one participant from the population draws without refitting.

# %%
from symphony_dt.data import Standardisation
from symphony_dt.forecast import forecast_fast

pid = ids[tests[0]]
history = cohort.dataset.subset([pid])
fast = forecast_fast(result, history, [1.0, 2.0], standardisation=Standardisation.fit(cohort.dataset),
                     n_particles=1000, n_draws=200, seed=1)
fast.quantiles()
