"""Shared fixtures. Slow tests compile Stan and run only with --run-slow."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False, help="run tests that fit Stan models")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def cohort():
    from symphony_dt.data import simulate_cohort

    return simulate_cohort(40, seed=11, n_pathway=2)


@pytest.fixture(scope="session")
def frozen_default():
    from symphony_dt.evaluate import ScoringSpec, freeze_spec

    spec = ScoringSpec.default()
    return spec, freeze_spec(spec)
