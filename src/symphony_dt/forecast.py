"""Forecasting: the exact route of the study and an experimental fast route.

Exact route. Mask each test participant's observations after an anchor (`split_at_anchor`), fit the
population model on the masked data, thin the chains (`thin_draws`) and run the generated-quantities
program on the masked targets (`forecast_exact`). The forecast is `y_rep_c`: the observation model
applied to the posterior latent path, whose post-anchor steps carry no data and are therefore draws
from the dynamics.

Fast route (EXPERIMENTAL). Condition a new participant's latent path on each population posterior
draw by importance sampling, with no refit (`forecast_fast`). It approximates the exact route in two
ways: the population draws do not learn from the new participant, and the participant's posterior
is represented by a finite set of weighted particles. Tests compare it with the exact route on
synthetic data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, log_ndtr, logsumexp, ndtr, ndtri

from symphony_dt import constants as C
from symphony_dt.config import ModelConfig
from symphony_dt.data.dataset import TwinDataset
from symphony_dt.data.stan_data import Standardisation, StanData, build_stan_data, substeps_days
from symphony_dt.diagnostics import chain_csvs_in_order
from symphony_dt.evaluate.metrics import event_probability
from symphony_dt.labels import validation

__all__ = [
    "Forecast",
    "HoldoutSplit",
    "forecast_exact",
    "forecast_fast",
    "global_draws_from_csv",
    "read_gq_csv",
    "run_standalone_gq",
    "split_at_anchor",
    "thin_draws",
]


@validation("TESTED")
@dataclass
class HoldoutSplit:
    """The fit data (test histories masked after the anchor), the target data and target metadata.

    Validation: TESTED.
    """

    fit: StanData
    targets: StanData
    meta: pd.DataFrame
    test_index: list[int]
    anchor_years: float

    def write(self, directory: str | Path) -> dict[str, Path]:
        """Write stan_data_fit.json, stan_data_targets.json, targets_meta.csv and test_index.json.

        Validation: TESTED.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "fit": self.fit.to_json(out / "stan_data_fit.json"),
            "targets": self.targets.to_json(out / "stan_data_targets.json"),
            "meta": out / "targets_meta.csv",
            "test_index": out / "test_index.json",
        }
        self.meta.to_csv(paths["meta"], index=False)
        paths["test_index"].write_text(json.dumps(sorted(self.test_index)))
        return paths


@validation("TESTED")
def split_at_anchor(
    stan_data: StanData | Mapping[str, Any],
    test_index: Sequence[int],
    *,
    anchor_years: float = 2.0,
    zero_dose_std: float = 0.0,
    cohort: Sequence[str] | None = None,
    sex: Sequence[int] | None = None,
) -> HoldoutSplit:
    """Mask every test participant's observations later than `anchor_years` after baseline.

    The step grid, covariates and every training observation are kept, so the latent path at a
    masked node is driven by the dynamics alone. Each masked observation becomes a target. A
    UPDRS-III target carries the participant's last kept UPDRS-III LEDD forward, or
    `zero_dose_std` when there is none. The anchor time of a participant is its last kept
    observation at or before the anchor on any endpoint, zero when there is none.

    This reimplements the study's `fit_holdout.split` from its written design; the original
    exists only on the study's cluster. Validation: TESTED.
    """
    d = dict(stan_data.data if isinstance(stan_data, StanData) else stan_data)
    tests = {int(i) for i in test_index}
    n = int(d["N"])
    if not tests.issubset(range(n)):
        raise ValueError("test_index holds an index outside 0 to N - 1")
    fit_arrays: dict[str, list[Any]] = {"obs_node": [], "obs_k": [], "y": [], "obs_ledd": []}
    tgt_arrays: dict[str, list[Any]] = {"obs_node": [], "obs_k": [], "y": [], "obs_ledd": []}
    fit_start, fit_count = np.zeros(n, int), np.zeros(n, int)
    tgt_start, tgt_count = np.zeros(n, int), np.zeros(n, int)
    meta_rows: list[dict[str, Any]] = []
    for i in range(n):
        start, k_steps = int(d["step_start"][i]), int(d["K"][i])
        tt = np.concatenate([[0.0], np.cumsum(np.asarray(d["h"][start : start + k_steps], float))])
        fit_start[i] = len(fit_arrays["y"])
        tgt_start[i] = len(tgt_arrays["y"])
        a, c = int(d["obs_start"][i]), int(d["obs_count"][i])
        kept_times: list[float] = []
        last_ledd: float | None = None
        pending: list[tuple[int, int, float, float]] = []
        for j in range(a, a + c):
            node, k, y = int(d["obs_node"][j]), int(d["obs_k"][j]), float(d["y"][j])
            t = float(tt[node - 1])
            if i in tests and t > anchor_years:
                pending.append((node, k, y, t))
                continue
            fit_arrays["obs_node"].append(node)
            fit_arrays["obs_k"].append(k)
            fit_arrays["y"].append(y)
            fit_arrays["obs_ledd"].append(float(d["obs_ledd"][j]))
            kept_times.append(t)
            if k == 1:
                last_ledd = float(d["obs_ledd"][j])
        anchor_t = max(kept_times) if kept_times else 0.0
        for node, k, y, t in pending:
            tgt_arrays["obs_node"].append(node)
            tgt_arrays["obs_k"].append(k)
            tgt_arrays["y"].append(y)
            ledd = (last_ledd if last_ledd is not None else zero_dose_std) if k == 1 else 0.0
            tgt_arrays["obs_ledd"].append(float(ledd))
            meta_rows.append(
                {
                    "i": i,
                    "k": k,
                    "node": node,
                    "t": t,
                    "anchor_t": anchor_t,
                    "horizon": t - anchor_t,
                    "y": y,
                    "cohort": str(cohort[i]) if cohort is not None else "",
                    "sex": int(sex[i]) if sex is not None else -1,
                }
            )
        fit_count[i] = len(fit_arrays["y"]) - fit_start[i]
        tgt_count[i] = len(tgt_arrays["y"]) - tgt_start[i]
    if not tgt_arrays["y"]:
        raise ValueError("no observation lies after the anchor for any test participant")
    fit = {
        **d,
        **fit_arrays,
        "n_obs": len(fit_arrays["y"]),
        "obs_start": fit_start.tolist(),
        "obs_count": fit_count.tolist(),
    }
    tgt = {
        **d,
        **tgt_arrays,
        "n_obs": len(tgt_arrays["y"]),
        "obs_start": tgt_start.tolist(),
        "obs_count": tgt_count.tolist(),
    }
    ids = stan_data.participant_ids if isinstance(stan_data, StanData) else []
    return HoldoutSplit(
        fit=StanData(fit, {"split": "fit", "anchor_years": anchor_years}, ids),
        targets=StanData(tgt, {"split": "targets", "anchor_years": anchor_years}, ids),
        meta=pd.DataFrame(meta_rows),
        test_index=sorted(tests),
        anchor_years=anchor_years,
    )


@validation("VALIDATED")
def thin_draws(csv_files: Sequence[str | Path], output: str | Path, *, total_keep: int = 400) -> Path:
    """Write one CmdStan CSV holding every THIN-th draw of every chain, THIN = draws x chains / keep.

    Keeps chain 1's comment header and adaptation block and the column header, which must be
    identical across chains. Raises when the kept total differs from `total_keep`.

    Validation: VALIDATED. Port of the study's `thin_fold.py`.
    """
    files = chain_csvs_in_order(csv_files)
    counts = []
    for p in files:
        with open(p) as handle:
            rows = [ln for ln in handle if not ln.startswith("#") and ln.strip()]
        counts.append(len(rows) - 1)
    if len(set(counts)) != 1:
        raise ValueError(f"chains hold unequal draws {counts}")
    thin = counts[0] * len(files) // total_keep
    if thin < 1:
        raise ValueError(f"cannot keep {total_keep} draws from {counts[0] * len(files)}")
    header0 = None
    kept = 0
    first = True
    out_path = Path(output)
    with out_path.open("w") as out:
        for p in files:
            header = None
            n = 0
            with open(p) as f:
                for line in f:
                    if header is None:
                        if line.startswith("#"):
                            if first:
                                out.write(line)
                            continue
                        header = line
                        if first:
                            header0 = header
                            out.write(line)
                        elif header != header0:
                            raise ValueError(f"column header differs in {p}")
                        continue
                    if line.startswith("#") or not line.strip():
                        if first and n == 0:
                            out.write(line)
                        continue
                    if n % thin == 0:
                        out.write(line)
                        kept += 1
                    n += 1
            first = False
    if kept != total_keep:
        raise ValueError(f"kept {kept} draws, expected {total_keep}; choose total_keep dividing the draws")
    return out_path


@validation("VALIDATED")
def read_gq_csv(path: str | Path, n_targets: int, prefix: str = "y_rep_c") -> NDArray[np.float64]:
    """Read the `prefix.1` to `prefix.n` columns of a generate_quantities CSV as (draws, targets).

    Raises when the columns do not match the target count and order or are not contiguous.

    Validation: VALIDATED. Port of the study's `read_gq`.
    """
    with open(path) as f:
        rows = (line for line in f if not line.startswith("#") and line.strip())
        header = next(rows).rstrip("\n").split(",")
        idx = [j for j, h in enumerate(header) if h.startswith(prefix + ".")]
        names = [header[j] for j in idx]
        if names != [f"{prefix}.{q + 1}" for q in range(n_targets)]:
            raise ValueError(f"{prefix} columns do not match the target count or order")
        lo, hi = idx[0], idx[-1] + 1
        if idx != list(range(lo, hi)):
            raise ValueError(f"{prefix} columns are not contiguous")
        draws = [np.asarray(line.rstrip("\n").split(",")[lo:hi], float) for line in rows]
    return np.vstack(draws)


@validation("TESTED")
@dataclass
class Forecast:
    """Forecast draws for a set of targets, with summaries and export.

    `draws` is (draws, targets). `targets` has one row per target with at least `k` (endpoint index)
    and usually `i`, `t`, `horizon` and `y`.

    Validation: TESTED.
    """

    draws: NDArray[np.float64]
    targets: pd.DataFrame
    method: str = "exact"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.draws = np.asarray(self.draws, float)
        if self.draws.ndim != 2 or self.draws.shape[1] != len(self.targets):
            raise ValueError(f"draws {self.draws.shape} do not match {len(self.targets)} targets")

    @property
    def n_draws(self) -> int:
        """Number of draws per target."""
        return int(self.draws.shape[0])

    def quantiles(self, q: Sequence[float] = (0.025, 0.25, 0.5, 0.75, 0.975)) -> pd.DataFrame:
        """Return the targets with one column per requested quantile.

        Validation: TESTED.
        """
        values = np.quantile(self.draws, list(q), axis=0).T
        cols = {f"q{str(v).replace('0.', '')}": values[:, j] for j, v in enumerate(q)}
        return self.targets.reset_index(drop=True).assign(**cols)

    def interval(self, level: float = 0.95) -> pd.DataFrame:
        """Return the targets with `lower` and `upper` of the central interval at `level`.

        Validation: TESTED.
        """
        tail = (1 - level) / 2
        lo, hi = np.quantile(self.draws, [tail, 1 - tail], axis=0)
        return self.targets.reset_index(drop=True).assign(lower=lo, upper=hi, level=level)

    def mean(self) -> NDArray[np.float64]:
        """Return the forecast mean per target.

        Validation: TESTED.
        """
        return np.asarray(self.draws.mean(axis=0), float)

    def event_probability(
        self, last_values: ArrayLike, thresholds: Mapping[int, tuple[str, float]] | None = None
    ) -> NDArray[np.float64]:
        """Return per-target event probabilities against each target's last observed value.

        Targets whose endpoint has no event rule get NaN. Defaults: +5 UPDRS-III, -2 MoCA.

        Validation: TESTED.
        """
        from symphony_dt.evaluate.metrics import EVENT_THRESHOLDS

        rules = thresholds or EVENT_THRESHOLDS
        last = np.asarray(last_values, float)
        out = np.full(len(self.targets), np.nan)
        for q, k in enumerate(self.targets["k"].to_numpy(int)):
            if k in rules and np.isfinite(last[q]):
                out[q] = event_probability(self.draws[:, q], float(last[q]), int(k), rules)
        return out

    def to_frame(self) -> pd.DataFrame:
        """Return the draws in long format: one row per target and draw.

        Validation: TESTED.
        """
        n_draw, n_tgt = self.draws.shape
        frame = pd.DataFrame(
            {
                "target": np.tile(np.arange(n_tgt), n_draw),
                "draw": np.repeat(np.arange(n_draw), n_tgt),
                "value": self.draws.ravel(),
            }
        )
        return frame.merge(
            self.targets.reset_index(drop=True).rename_axis("target").reset_index(), on="target", how="left"
        )

    def to_csv(self, path: str | Path) -> Path:
        """Write the quantile table (not the draws) to CSV.

        Validation: TESTED.
        """
        target = Path(path)
        self.quantiles().to_csv(target, index=False)
        return target

    def to_netcdf(self, path: str | Path) -> Path:
        """Write the draws and targets to netCDF via xarray.

        Validation: TESTED.
        """
        import xarray as xr

        ds = xr.Dataset(
            {"forecast": (("draw", "target"), self.draws)},
            coords={"draw": np.arange(self.n_draws), "target": np.arange(len(self.targets))},
        )
        for col in self.targets.columns:
            values = self.targets[col].to_numpy()
            ds = ds.assign_coords(
                {f"target_{col}": ("target", values.astype(str) if values.dtype == object else values)}
            )
        ds.attrs.update({"method": self.method, "producer": "symphony-dt"})
        target = Path(path)
        ds.to_netcdf(target, engine="h5netcdf")
        return target

    def plot(self, participant: int | None = None, ax: Any = None) -> Any:
        """Plot forecast intervals by horizon; see `symphony_dt.plots.plot_forecast`.

        Validation: TESTED.
        """
        from symphony_dt.plots import plot_forecast

        return plot_forecast(self, participant=participant, ax=ax)


@validation("TESTED")
def run_standalone_gq(
    exe_file: str | Path, data: str | Path, fitted_params: str | Path, output_csv: str | Path, *, seed: int
) -> Path:
    """Run CmdStan's standalone generate_quantities method directly on a fitted-parameters CSV.

    The binary is called without cmdstanpy's CSV validation, because a thinned file pooled over
    chains carries no per-chain timing footer, which CmdStan itself does not need.

    Validation: TESTED.
    """
    import os
    import subprocess

    out = Path(output_csv)
    cmd = [
        str(exe_file),
        "generate_quantities",
        f"fitted_params={fitted_params}",
        "data",
        f"file={data}",
        "output",
        f"file={out}",
        "random",
        f"seed={seed}",
    ]
    env = {**os.environ, "STAN_NUM_THREADS": "1"}
    done = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"generate_quantities failed: {(done.stderr or done.stdout)[-800:]}")
    return out


@validation("TESTED")
def forecast_exact(
    fit: Any,
    split: HoldoutSplit,
    *,
    total_keep: int = 400,
    seed: int = 20261021,
    output_dir: str | Path | None = None,
    model: str = "pd_v2",
) -> Forecast:
    """Forecast the masked targets from a fit of `split.fit` with the generated-quantities program.

    Thins the chains to `total_keep` draws, runs standalone generate_quantities on the target data
    with `seed`, and returns the `y_rep_c` draws. `fit` is a FitResult of `split.fit`.

    Validation: TESTED. The route is the study's design; the GQ program is the package's rewrite.
    """
    from symphony_dt.models import compile_model

    out = Path(output_dir) if output_dir is not None else Path(fit.output_dir) / "forecast"
    out.mkdir(parents=True, exist_ok=True)
    thin = thin_draws(fit.csv_files, out / "thin.csv", total_keep=total_keep)
    data_path = split.targets.to_json(out / "stan_data_targets.json")
    exe = compile_model(model, "forecast_gq")
    csv = run_standalone_gq(exe.exe_file, data_path, thin, out / "gq.csv", seed=seed)
    draws = read_gq_csv(csv, int(split.targets.data["n_obs"]))
    return Forecast(
        draws=draws,
        targets=split.meta.copy(),
        method="exact",
        metadata={"thin_csv": str(thin), "gq_csv": str(csv), "seed": seed, "total_keep": total_keep},
    )


_GLOBAL_SHAPES: dict[str, tuple[int, ...]] = {
    "mu_r": (3,),
    "Gamma": (3, 3),
    "sigma_r": (3,),
    "sigma_xi": (3,),
    "a": (3,),
    "b": (3,),
    "sigma_obs": (3,),
    "kappa": (3,),
    "lambda_ledd": (),
    "c_MM": (),
    "c_CM": (),
    "c_CC": (),
    "c_CN": (),
    "c_NM": (),
    "c_NN": (),
}


@validation("TESTED")
def global_draws_from_csv(
    csv_files: Sequence[str | Path], n_pathway: int = 0, max_draws: int | None = None
) -> list[dict[str, Any]]:
    """Read the global parameters of a pd_v2 fit as a list of parameter dicts, one per draw.

    Validation: TESTED.
    """
    shapes = {**_GLOBAL_SHAPES, "Amat": (3, n_pathway)}
    frames = [pd.read_csv(p, comment="#") for p in chain_csvs_in_order(csv_files)]
    frame = pd.concat(frames, ignore_index=True)
    if max_draws is not None and len(frame) > max_draws:
        frame = frame.iloc[np.linspace(0, len(frame) - 1, max_draws).round().astype(int)]
    out: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        theta: dict[str, Any] = {}
        for name, shape in shapes.items():
            if shape == ():
                theta[name] = float(row[name])
            elif len(shape) == 1:
                theta[name] = np.array([row[f"{name}.{j + 1}"] for j in range(shape[0])], float)
            else:
                theta[name] = np.array(
                    [[row[f"{name}.{r + 1}.{c + 1}"] for c in range(shape[1])] for r in range(shape[0])],
                    float,
                ).reshape(shape)
        out.append(theta)
    return out


def _endpoint_logpdf(
    y: float, mu: NDArray[np.float64], sd: NDArray[np.float64], lo: float, hi: float, censored: int
) -> NDArray[np.float64]:
    z = (y - mu) / sd
    logpdf = -0.5 * z * z - np.log(sd) - 0.5 * np.log(2 * np.pi)
    if censored:
        if y >= hi:
            return np.asarray(log_ndtr(-(hi - mu) / sd), float)
        if y <= lo:
            return np.asarray(log_ndtr((lo - mu) / sd), float)
        return np.asarray(logpdf, float)
    mass = np.clip(ndtr((hi - mu) / sd) - ndtr((lo - mu) / sd), 1e-300, None)
    return np.asarray(logpdf - np.log(mass), float)


def _endpoint_rng(
    mu: NDArray[np.float64],
    sd: NDArray[np.float64],
    lo: float,
    hi: float,
    censored: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    if censored:
        return np.asarray(np.clip(rng.normal(mu, sd), lo, hi), float)
    pl, pu = ndtr((lo - mu) / sd), ndtr((hi - mu) / sd)
    u = np.clip(pl + rng.uniform(size=mu.shape) * (pu - pl), 1e-12, 1 - 1e-12)
    return np.asarray(np.clip(mu + sd * ndtri(u), lo, hi), float)


@validation("EXPERIMENTAL")
def forecast_fast(
    population: Any,
    history: TwinDataset,
    horizons_years: Sequence[float],
    *,
    standardisation: Standardisation | None = None,
    config: ModelConfig | None = None,
    n_particles: int = 2000,
    n_draws: int = 400,
    seed: int = 0,
    ledd_std: float | None = None,
    pathway: Sequence[str] = (),
) -> Forecast:
    """Forecast one new participant from population draws by importance sampling, with no refit.

    `population` is a FitResult, a list of chain CSV paths, or a list of parameter dicts.
    `history` holds one participant. For each population draw, `n_particles` latent paths are drawn
    from the prior, weighted by the likelihood of the history and one is resampled; it is then
    propagated `horizons_years` beyond the last visit with fresh innovations. `standardisation` must
    be the one used for the population fit. LEDD at the horizons is the last visit's value unless
    `ledd_std` is given.

    Validation: EXPERIMENTAL. An approximation of the exact route, compared with it in tests.
    """
    cfg = config or ModelConfig.primary()
    if len(history.participants) != 1:
        raise ValueError("history must hold exactly one participant")
    sd = build_stan_data(history, config=cfg, standardisation=standardisation, pathway=pathway)
    d = sd.data
    if isinstance(population, list) and population and isinstance(population[0], dict):
        thetas = population
    else:
        files = population.csv_files if hasattr(population, "csv_files") else population
        thetas = global_draws_from_csv(files, n_pathway=int(d["P"]), max_draws=n_draws)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(thetas), size=n_draws, replace=len(thetas) < n_draws)
    lower = np.asarray(d["Lk"], float)
    upper = np.asarray(d["Uk"], float)
    censored = list(d["censored"])
    K = int(d["K"][0])
    h = np.asarray(d["h"], float)
    d_i = np.asarray(d["dmat"], float)[:, 0]
    z_i = np.asarray(d["zmat"], float)[:, 0] if int(d["P"]) else np.zeros(0)
    obs_node = np.asarray(d["obs_node"], int)
    obs_k = np.asarray(d["obs_k"], int)
    y = np.asarray(d["y"], float)
    obs_ledd = np.asarray(d["obs_ledd"], float)
    updrs = obs_ledd[obs_k == 1]
    fc_ledd = float(ledd_std) if ledd_std is not None else (float(updrs[-1]) if updrs.size else 0.0)
    xbar = np.asarray(d["sigma_xbar"], float)
    horizon_days = [int(round(v * C.DAYS_PER_YEAR)) for v in horizons_years]
    sub = substeps_days(np.asarray(horizon_days, float), cfg.h_max_years)
    out = np.zeros((n_draws, 3 * len(horizons_years)))
    ess: list[float] = []

    def sigma(theta: Mapping[str, Any], k: int, s_k: NDArray[np.float64]) -> NDArray[np.float64]:
        base = np.full_like(s_k, float(theta["sigma_obs"][k]))
        if int(d["state_dependent_sigma"]):
            base = base * np.exp(float(theta["kappa"][k]) * (expit(s_k) - xbar[k]))
        return np.maximum(base, 1e-6)

    def mean(theta: Mapping[str, Any], k: int, s_k: NDArray[np.float64], ledd: float) -> NDArray[np.float64]:
        lin = float(theta["a"][k]) + float(theta["b"][k]) * s_k
        if int(d["ledd_in_mu_U"]) and k == 0:
            lin = lin + float(theta["lambda_ledd"]) * ledd
        link = expit(lin)
        return np.asarray(lower[k] + (upper[k] - lower[k]) * ((1 - link) if k == 1 else link), float)

    def step(
        s: NDArray[np.float64],
        r: NDArray[np.float64],
        Cm: NDArray[np.float64],
        sig_xi: NDArray[np.float64],
        dt: float,
        e: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        drive = np.logaddexp(0.0, r + expit(s) @ Cm.T)
        xi = np.exp(-0.5 * dt * sig_xi**2 + np.sqrt(dt) * sig_xi * e)
        return np.asarray(s + dt * drive * xi, float)

    from symphony_dt.data.simulate import coupling_matrix

    for row, idx in enumerate(pick):
        theta = thetas[int(idx)]
        Cm = coupling_matrix(theta)
        sig_xi = np.asarray(theta["sigma_xi"], float)
        amat = np.asarray(theta.get("Amat", np.zeros((3, 0))), float)
        base_r = (
            np.asarray(theta["mu_r"], float)
            + np.asarray(theta["Gamma"], float) @ d_i
            + (amat @ z_i if z_i.size else 0.0)
        )
        r = base_r + np.asarray(theta["sigma_r"], float) * rng.normal(size=(n_particles, 3))
        s = rng.normal(size=(n_particles, 3))
        nodes = [s]
        for kk in range(K):
            s = step(s, r, Cm, sig_xi, float(h[kk]), rng.normal(size=(n_particles, 3)))
            nodes.append(s)
        logw = np.zeros(n_particles)
        for node, k, value, ledd in zip(obs_node, obs_k, y, obs_ledd, strict=True):
            s_k = nodes[node - 1][:, k - 1]
            logw += _endpoint_logpdf(
                value,
                mean(theta, k - 1, s_k, ledd),
                sigma(theta, k - 1, s_k),
                lower[k - 1],
                upper[k - 1],
                censored[k - 1],
            )
        w = np.exp(logw - logsumexp(logw))
        ess.append(float(1.0 / np.sum(w**2)))
        chosen = rng.choice(n_particles, p=w)
        s_last = nodes[-1][chosen : chosen + 1]
        r_c = r[chosen : chosen + 1]
        for f, (days, count) in enumerate(zip(horizon_days, sub, strict=True)):
            sf = s_last.copy()
            if count:
                dt = days / C.DAYS_PER_YEAR / count
                for _ in range(int(count)):
                    sf = step(sf, r_c, Cm, sig_xi, dt, rng.normal(size=(1, 3)))
            for k in range(3):
                mu = mean(theta, k, sf[:, k], fc_ledd)
                out[row, 3 * f + k] = _endpoint_rng(
                    mu, sigma(theta, k, sf[:, k]), lower[k], upper[k], censored[k], rng
                )[0]
    targets = pd.DataFrame(
        [
            {
                "k": k + 1,
                "endpoint": C.ENDPOINTS[k],
                "horizon": float(hz),
                "participant_id": sd.participant_ids[0],
            }
            for hz in horizons_years
            for k in range(3)
        ]
    )
    return Forecast(
        draws=out,
        targets=targets,
        method="fast",
        metadata={
            "n_particles": n_particles,
            "n_draws": n_draws,
            "seed": seed,
            "median_ess": float(np.median(ess)),
            "min_ess": float(np.min(ess)),
        },
    )
