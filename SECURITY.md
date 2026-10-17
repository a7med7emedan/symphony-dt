# Security policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.2.x   | Yes       |
| < 0.2   | No        |

Fixes are made on the latest 0.2.x release.

## Reporting a vulnerability

Please report security problems privately through GitHub:
<https://github.com/a7med7emedan/symphony-dt/security/advisories/new>

Do not open a public issue, pull request or discussion for a security problem.
Include what you found, how to reproduce it, and the version or commit affected.
You should receive an acknowledgement within seven days.

## Participant data

This repository contains no PPMI data and no other participant-level data.
The code, tests, examples and container images use synthetic tables only
(see `symphony-dt synth`).

If you find participant data, or anything derived from it, in this repository,
its history, a release, a container image or an issue, treat it as a security
issue. Report it privately through the link above rather than in public, so the
data can be removed and the exposure assessed without spreading it further.
