from typer.testing import CliRunner

from symphony_dt import __version__
from symphony_dt.cli import app


def test_version_flag_reports_installed_version():
    r = CliRunner().invoke(app, ["--version"])
    assert r.exit_code == 0
    assert __version__ in r.output
    assert __version__ != "0.0.0+unknown"
