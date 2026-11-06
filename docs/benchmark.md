# Benchmark

`symphony-dt bench` runs five synthetic scenarios with known truth and scores every method with the frozen
code-of-record specification.

| scenario | what changes |
|---|---|
| dense | visits every six months to a year |
| sparse | few visits, one to two years apart |
| floor | motor scores pushed toward zero |
| dropout | a participant leaves after the motor score passes 40 |
| drift | from year three, motor scores read three points high |

The twin is the living twin on the generating parameters, an oracle population. This isolates the
participant-level machinery and the scoring from population estimation. Pass your own population draws to
`run_benchmark` to include estimation. Baselines are LOCF and the linear mixed model of the scoring code.

Run `symphony-dt bench --out benchmark.csv` to reproduce the table on your machine. These are synthetic results.
They say nothing about performance on any clinical cohort.
