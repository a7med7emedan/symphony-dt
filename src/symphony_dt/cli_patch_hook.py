"""Module 24 hook to register new CLI commands.

Add this single import line near the top of symphony_dt/cli.py:

    import symphony_dt.cli_patch_hook  # noqa: F401

This registers:
  - aggregate-all
  - parity-check
"""

from symphony_dt.cli import app  # noqa: F401
from symphony_dt.cli_register import register_module24_commands

register_module24_commands(app)
