# syntax=docker/dockerfile:1
# symphony-dt container. Ships code only: no PPMI data, no results, no credentials.
# Mount your own preprocessed tables at runtime, or run the synthetic quickstart.

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set INFERENCE=1 to include NumPyro/JAX for the `nuts` commands.
ARG INFERENCE=0

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin symphony
WORKDIR /opt/symphony-dt

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY configs ./configs

RUN if [ "$INFERENCE" = "1" ]; then pip install ".[inference]"; else pip install .; fi \
    && rm -rf /root/.cache

USER symphony
WORKDIR /work

ENTRYPOINT ["symphony-dt"]
CMD ["--help"]
