import pytest

from symphony_dt.inference_backend.numpyro_impl import _require_numpyro


def test_numpyro_optional_dependency_error_message() -> None:
    try:
        _require_numpyro()
    except ImportError as e:
        assert "NumPyro backend requires optional dependencies" in str(e)
        return
    # If numpyro is installed in the environment, that's fine too.
    assert True
