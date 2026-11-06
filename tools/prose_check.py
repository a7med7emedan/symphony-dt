#!/usr/bin/env python3
"""Prose check for the documentation, README and docstrings.

Rule D: no em or en dash anywhere in prose or code we write (the frozen Stan files excepted).
Rule S: in Markdown prose, no sentence of 25 words or more. Code blocks, tables, headings, HTML
and quotations are skipped. Exit status 1 on any finding.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

DASHES = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")
SKIP_PREFIX = ("STALE_", ".dev/", "tests/data/")
SKIP_FILES = {"src/symphony_dt/models/pd_v2/model.stan", "src/symphony_dt/models/pd_v2/conditional.stan"}
TEXT_SUFFIX = {".md", ".py", ".yaml", ".yml", ".toml", ".cff", ".json", ".stan", ".txt", ".ipynb"}
LIMIT = 25


def files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [
        Path(p)
        for p in out
        if not p.startswith(SKIP_PREFIX) and p not in SKIP_FILES and Path(p).suffix in TEXT_SUFFIX and Path(p).exists()
    ]


def sentences(markdown: str) -> list[tuple[int, str]]:
    """Return (first line number, sentence) for prose paragraphs, joining wrapped lines."""
    found: list[tuple[int, str]] = []
    in_code = False
    paragraph: list[str] = []
    start = 0

    def flush() -> None:
        if not paragraph:
            return
        text = " ".join(paragraph)
        text = re.sub(r"`[^`]*`", "code", text)
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"\\\(.*?\\\)", "math", text)
        text = re.sub(r"\b(e\.g|i\.e|et al|cf|vs)\.", r"\1", text)
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            found.append((start, sentence))
        paragraph.clear()

    for number, line in enumerate(markdown.splitlines(), 1):
        stripped = line.strip()
        if line.startswith("    ") and not paragraph:
            flush()
            continue
        if stripped.startswith("```"):
            flush()
            in_code = not in_code
            continue
        skip = in_code or not stripped or stripped.startswith(("#", "|", "<", ">", "!!!", "::: ", "["))
        if skip:
            flush()
            continue
        if stripped.startswith(("- ", "* ")) or re.match(r"\d+\. ", stripped):
            flush()
            stripped = re.sub(r"^(- |\* |\d+\. )", "", stripped)
        if not paragraph:
            start = number
        paragraph.append(stripped)
    flush()
    return found


def main() -> int:
    findings: list[str] = []
    for path in files():
        text = path.read_text(errors="ignore")
        for number, line in enumerate(text.splitlines(), 1):
            if DASHES.search(line):
                findings.append(f"D {path}:{number} em or en dash")
        if path.suffix == ".md" and (
            str(path).startswith("docs/") or path.name in {"README.md", "GOVERNANCE.md", "CONTRIBUTING.md"}
        ):
            for number, sentence in sentences(text):
                words = len(sentence.split())
                if words >= LIMIT:
                    findings.append(f"S {path}:{number} sentence of {words} words: {sentence[:60]}...")
    for f in findings:
        print(f"PROSE: {f}")
    print(f"prose check: {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
