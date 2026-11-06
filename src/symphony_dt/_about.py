"""The `about` banner: logo, version, author, citation and documentation address."""

from __future__ import annotations

import sys
from typing import Any, TextIO

from symphony_dt._version import __version__
from symphony_dt.labels import validation

__all__ = ["ASCII_LOGO", "CITATION", "DOCS_URL", "about"]

DOCS_URL = "https://a7med7emedan.github.io/symphony-dt/"

#: The SYMPHONY tulip: the solid half is the patient, the dotted half is the digital twin.
ASCII_LOGO = r"""
            /\    /\    /\
           /##\  /#|:\  /::\
          |####\/##|::\/::::|
          |########|::::::::|
           \#######|:::::::/
            '.#####|:::::.'
              '-.##|::.-'
         .--.      |      .:.
        (####'-.   |   .-'::)
         '--.__'-. | .-'__:'
                '-.|.-'
                   |
          s y m p h o n y - d t
"""

CITATION = (
    "Hemedan A. symphony-dt: Bayesian digital twins of Parkinson's disease progression, "
    f"version {__version__}. https://github.com/a7med7emedan/symphony-dt"
)


@validation("TESTED")
def about(file: TextIO | Any = None) -> str:
    """Print the logo, version, author, citation and documentation address, and return the text.

    Pass `file` to print elsewhere than standard output.

    Validation: TESTED.
    """
    text = "\n".join(
        [
            ASCII_LOGO.rstrip("\n"),
            "",
            f"  symphony-dt {__version__}",
            "  Developed by Ahmed Hemedan",
            "",
            f"  Cite: {CITATION}",
            "  Validation manuscript in preparation.",
            f"  Docs: {DOCS_URL}",
            "",
        ]
    )
    print(text, file=file if file is not None else sys.stdout)
    return text
