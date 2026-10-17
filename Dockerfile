# syntax=docker/dockerfile:1
# symphony-dt container. Ships code only: no PPMI data, no results, no credentials.
# Mount your own preprocessed tables at runtime, or run the synthetic quickstart.
#
# Build:  docker build -t symphony-dt .
#         docker build --build-arg INFERENCE=1 -t symphony-dt:inference .
#
# The base image is pinned by digest so builds are repeatable; Dependabot proposes
# digest updates within Python 3.12.

# ---- builder: turn the source tree into a wheel -------------------------------
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip wheel --no-deps --wheel-dir /wheels .

# ---- runtime: install the wheel only ------------------------------------------
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS runtime

# Set INFERENCE=1 to include NumPyro/JAX for the `nuts` commands.
ARG INFERENCE=0
ARG VERSION=0.2.0

LABEL org.opencontainers.image.title="symphony-dt" \
      org.opencontainers.image.description="Bayesian digital twin for Parkinson's disease progression. Code only, no participant data." \
      org.opencontainers.image.source="https://github.com/a7med7emedan/symphony-dt" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin symphony \
    && mkdir -p /work \
    && chown symphony:symphony /work

# The wheel is bind-mounted from the builder, so it never lands in an image layer.
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    set -eu; \
    whl="$(ls /wheels/symphony_dt-*.whl)"; \
    if [ "$INFERENCE" = "1" ]; then pip install "${whl}[inference]"; else pip install "${whl}"; fi; \
    rm -rf /root/.cache

# Example configs stay at a fixed path, e.g. /opt/symphony-dt/configs/config.demo.yaml.
WORKDIR /opt/symphony-dt
COPY LICENSE NOTICE ./
COPY configs ./configs

USER symphony
WORKDIR /work

ENTRYPOINT ["symphony-dt"]
CMD ["--help"]
