# The symphony-dt image: the released wheel from PyPI, CmdStan 2.39.0 and pd_v2 compiled at build time.
# Published to ghcr.io/a7med7emedan/symphony-dt by .github/workflows/container.yml.
# Build: docker build --build-arg VERSION=2.1.0 -t symphony-dt .
# Run:   docker run --rm -v "$PWD":/work symphony-dt check
FROM python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54

ARG VERSION
ARG CMDSTAN_VERSION=2.39.0

LABEL org.opencontainers.image.title="symphony-dt" \
      org.opencontainers.image.description="A Bayesian digital twin of Parkinson's disease progression, with CmdStan and the pd_v2 model compiled." \
      org.opencontainers.image.source="https://github.com/a7med7emedan/symphony-dt" \
      org.opencontainers.image.documentation="https://a7med7emedan.github.io/symphony-dt/" \
      org.opencontainers.image.licenses="BSD-3-Clause" \
      org.opencontainers.image.authors="Ahmed Hemedan" \
      org.opencontainers.image.version="${VERSION}"

# A C++ toolchain stays in the image so that new Stan programs can still be compiled.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential make curl \
    && rm -rf /var/lib/apt/lists/*

RUN test -n "${VERSION}" \
    && pip install --no-cache-dir --retries 10 --timeout 60 "symphony-dt[io]==${VERSION}"

ENV CMDSTAN=/opt/cmdstan/cmdstan-${CMDSTAN_VERSION} \
    SYMPHONY_DT_CACHE=/opt/symphony-cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONDONTWRITEBYTECODE=1

# The threaded precompiled header serves pd_v2; the unthreaded one is rebuilt on demand, which saves about 1 GB.
RUN python -m cmdstanpy.install_cmdstan --version "${CMDSTAN_VERSION}" --dir /opt/cmdstan --cores "$(nproc)" \
    && symphony-dt check --compile \
    && find "${CMDSTAN}/stan/src/stan/model" -name 'model_header_[0-9]*.hpp.gch' -delete \
    && chmod -R a+rwX /opt/symphony-cache "${CMDSTAN}/stan/src/stan/model/model_header.hpp.gch"

# Run as an unprivileged user; any host user id also works with --user "$(id -u):$(id -g)".
RUN useradd --create-home --uid 1000 twin \
    && mkdir -p /work && chown twin:twin /work
USER twin
WORKDIR /work
ENTRYPOINT ["symphony-dt"]
CMD ["about"]
