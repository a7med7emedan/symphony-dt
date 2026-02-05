import pytest

pytest.importorskip("numpyro", reason="numpyro not installed (optional inference extras)")

def test_nuts_cli_imports() -> None:
    import symphony_dt.nuts_cli  # noqa: F401
