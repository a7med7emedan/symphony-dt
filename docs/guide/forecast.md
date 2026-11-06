# Forecasting

## Exact route

The exact route is the study's design. Mask each test participant's visits after an anchor, fit the population
model on the rest, and forecast the masked visits with the generated-quantities program.

```python
from symphony_dt.forecast import split_at_anchor, forecast_exact
split = split_at_anchor(stan_data, test_index, anchor_years=2.0)
result = fit(split.fit, output_dir="fit")
forecast = forecast_exact(result, split, total_keep=400, seed=20261021)
```

`forecast_exact` thins the chains to 400 draws and runs CmdStan's standalone generate_quantities on the targets.
The forecast is `y_rep_c`: the observation model on the posterior latent path. After the anchor the path carries
no data, so it is a draw from the dynamics. The package's GQ program is a rewrite of the study's design and is
labelled TESTED.

## Fast route

`forecast_fast(population, history, horizons_years)` forecasts a new participant without refitting. For each
population draw it samples latent paths from the prior, weights them by the likelihood of the participant's
history and resamples one. The route is EXPERIMENTAL. It approximates the exact route in two ways. The population
draws do not learn from the new person. A finite set of particles stands in for the person's posterior. The
metadata reports the effective sample size of the weights; low values call for more particles.

## The Forecast object

`Forecast` holds draws and target metadata. It offers `quantiles`, `interval`, `event_probability`, `to_frame`,
`to_csv`, `to_netcdf` and `plot`.
