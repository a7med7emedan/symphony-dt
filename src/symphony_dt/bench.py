"""An open benchmark of synthetic scenarios with known truth.

Each scenario simulates a cohort from the pd_v2 generative model, alters it in one named way, holds out every
test participant's visits after a two-year anchor and scores forecasts with the frozen code-of-record
specification. The twin is the living twin of `symphony_dt.twin`, run on the generating parameters (an oracle
population) unless a population is supplied, so the benchmark isolates the participant-level machinery and the
scoring. Baselines are LOCF and the linear mixed model of the scoring code.

Scenarios:
* `dense`: visits every six months to a year.
* `sparse`: few visits, one to two years apart.
* `floor`: motor scores pushed toward zero, where the truncated density matters.
* `dropout`: informative loss to follow-up; a participant leaves after their motor score passes 40.
* `drift`: from year three, every motor score reads three points high, a measurement drift the model does not know.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from symphony_dt import constants as C
from symphony_dt.data.dataset import TwinDataset
from symphony_dt.data.simulate import simulate_cohort, synthetic_visit_patterns
from symphony_dt.data.stan_data import Standardisation, build_stan_data
from symphony_dt.labels import validation

__all__ = ["SCENARIOS", "run_benchmark", "scenario_cohort"]

#: The benchmark scenarios, in order.
SCENARIOS: tuple[str, ...] = ("dense", "sparse", "floor", "dropout", "drift")


@validation("EXPERIMENTAL")
def scenario_cohort(name: str, n: int = 120, seed: int = 0) -> tuple[TwinDataset, dict[str, Any]]:
    """Simulate the cohort of one scenario; return the dataset and the generating parameters.

    Validation: EXPERIMENTAL.
    """
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; known: {list(SCENARIOS)}")
    rng = np.random.default_rng(seed)
    patterns = None
    if name == "dense":
        patterns = synthetic_visit_patterns(300, rng, mean_extra_visits=8, interval_days=(182, 365))
    elif name == "sparse":
        patterns = synthetic_visit_patterns(300, rng, mean_extra_visits=2, interval_days=(365, 730, 730))
    base = simulate_cohort(n, seed=seed, patterns=patterns)
    theta = dict(base.theta)
    if name == "floor":
        theta["a"] = np.asarray(theta["a"], float) - np.array([1.5, 0.0, 0.0])
        base = simulate_cohort(n, seed=seed, patterns=patterns, theta=theta)
    ds = base.dataset
    obs = ds.observations.copy()
    if name == "dropout":
        up = obs[obs["endpoint"] == "updrs3"].sort_values("time_days")
        leave = up[up["value"] > 40].groupby("participant_id")["time_days"].min()
        cut = obs["participant_id"].map(leave)
        obs = obs[cut.isna() | (obs["time_days"] <= cut)]
    if name == "drift":
        late = (obs["endpoint"] == "updrs3") & (obs["time_days"] >= 3 * C.DAYS_PER_YEAR)
        obs.loc[late, "value"] = np.clip(obs.loc[late, "value"] + 3.0, 0, 132)
    visits = ds.visits
    if visits is not None:
        keep = obs.groupby("participant_id")["time_days"].max()
        visits = visits[visits["time_days"] <= visits["participant_id"].map(keep).fillna(0)]
    out = TwinDataset(ds.participants, obs, visits, ds.markers, metadata={"scenario": name, "seed": seed})
    return out, theta


@validation("EXPERIMENTAL")
def run_benchmark(
    scenarios: Sequence[str] = SCENARIOS,
    *,
    n: int = 120,
    n_test: int = 40,
    seed: int = 0,
    population: Any = None,
    n_particles: int = 1000,
    n_samples: int = 400,
) -> pd.DataFrame:
    """Run the benchmark and return a tidy table of scores per scenario, endpoint, window, method and metric.

    `population` replaces the oracle with population draws (a PopulationDraws), for example from a fit.

    Validation: EXPERIMENTAL.
    """
    from symphony_dt.evaluate import ScoringSpec, freeze_spec, score_fold
    from symphony_dt.forecast import split_at_anchor
    from symphony_dt.outputs import scores_table
    from symphony_dt.twin import ParticipantTwin, PopulationDraws

    spec = ScoringSpec.default()
    frozen = freeze_spec(spec)
    tables = []
    for s_idx, name in enumerate(scenarios):
        ds, theta = scenario_cohort(name, n=n, seed=seed + 101 * s_idx)
        std = Standardisation.fit(ds)
        sd = build_stan_data(ds, standardisation=std)
        ids = sd.participant_ids
        part = ds.participants.set_index("participant_id").loc[ids]
        tests = sorted(
            np.random.default_rng(seed + s_idx).choice(len(ids), size=min(n_test, len(ids) - 1), replace=False).tolist()
        )
        split = split_at_anchor(
            sd,
            tests,
            anchor_years=spec.anchor_years,
            cohort=part["cohort"].tolist(),
            sex=part["sex"].astype(int).tolist(),
        )
        pop = population or PopulationDraws.from_thetas([theta], source=f"oracle:{name}")
        draws = np.zeros((n_samples, len(split.meta)))
        anchor_days = spec.anchor_years * C.DAYS_PER_YEAR
        for i_key, rows in split.meta.groupby("i"):
            i: int = int(str(i_key))
            pid = ids[i]
            hist = ds.subset([pid])
            hist.observations = hist.observations[hist.observations["time_days"] <= anchor_days]
            if hist.visits is not None:
                hist.visits = hist.visits[hist.visits["time_days"] <= anchor_days]
            twin = ParticipantTwin.create(pop, hist, standardisation=std, n_particles=n_particles, seed=seed + i)
            now = (twin.time_days or 0) / C.DAYS_PER_YEAR
            horizons = sorted({round(float(t) - now, 6) for t in rows["t"]})
            fc = twin.forecast(horizons, n_samples=n_samples, seed=seed + i)
            lookup = {
                (round(h, 6), k): col
                for col, (h, k) in enumerate(zip(fc.targets["horizon"], fc.targets["k"], strict=True))
            }
            for q, row in zip(rows.index.tolist(), rows.to_dict(orient="records"), strict=True):
                draws[:, q] = fc.draws[:, lookup[(round(float(row["t"]) - now, 6), int(row["k"]))]]
        scores = score_fold(
            split.fit.data,
            split.targets.data,
            split.meta,
            draws,
            tests,
            spec=spec,
            frozen=frozen,
            age=part["age_baseline_years"].astype(float).tolist(),
        )
        table = scores_table(scores.scores)
        table.insert(0, "scenario", name)
        table["population"] = pop.source
        tables.append(table)
    return pd.concat(tables, ignore_index=True)
