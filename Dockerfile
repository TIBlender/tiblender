# Digest-pinned base image for reproducible builds.
# python:3.12-slim, linux/amd64, verified with `docker manifest inspect python:3.12-slim`.
FROM python:3.12-slim@sha256:2fe5997d249a808b8eeea52c58a1dbffbba28754dc11699ef5c029f2d818ce79

WORKDIR /app

# Deterministic-by-default runtime: fixed hash seed, unbuffered stdio, no
# bytecode cache, offline pipeline mode with the pinned demo seed and a fixed
# clock. Any of these can still be overridden at `docker run`/compose time.
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIBLENDER_MODE=offline \
    TIBLENDER_SEED=20260909 \
    TIBLENDER_CLOCK=auto

# Chromium is installed as root but has to be readable by the unprivileged
# runtime user, so it goes to a shared, world-readable location instead of
# root's ~/.cache/ms-playwright. This must be set BEFORE `playwright install`
# below, or the browser lands in root's home and is unreachable at runtime.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optional Chromium browser for TIBLENDER_WEB_CRAWL_RENDER_JS=true. Disable
# with `--build-arg INSTALL_BROWSER=false` for smaller, faster offline images
# that only need static-HTML web crawling.
#
# `playwright install --with-deps chromium` cannot be used as-is on this
# digest-pinned base: playwright==1.50.0 (Jan 2025) predates Debian 13
# "trixie" (the OS this python:3.12-slim digest resolves to) and falls back
# to an ubuntu20.04 apt package list that references two packages Debian
# trixie no longer ships (`ttf-ubuntu-font-family`, dropped outright; and
# `ttf-unifont`, renamed `fonts-unifont`), so the install hard-fails with
# "has no installation candidate". The apt list below is the same
# `playwright install-deps chromium --dry-run` output with only those two
# substitutions, installed manually before a plain (no --with-deps)
# `playwright install chromium`; verified end-to-end (launch + render) in a
# standalone container against this exact base image digest.
#
# Reproducibility: trixie is Debian's rolling "testing" archive, so a plain
# `apt-get install` here would silently drift to whatever versions are
# current on deb.debian.org at build time. Pin apt to a dated
# snapshot.debian.org mirror instead. DEBIAN_SNAPSHOT defaults to
# 20260824T000000Z -- the exact snapshot this base image's own
# /etc/apt/sources.list.d/debian.sources already cites in a comment as the
# source of its own installed packages (confirmed present on both
# snapshot.debian.org/archive/debian and .../debian-security). Bump it with
# `--build-arg DEBIAN_SNAPSHOT=<YYYYMMDDTHHMMSSZ>` to move to newer packages
# without losing reproducibility; `Check-Valid-Until: no` is required because
# snapshot Release files carry a short Valid-Until that would otherwise make
# older snapshots unusable by the time this is rebuilt.
#
# If snapshot.debian.org is unreachable at build time (offline/air-gapped
# build host), `apt-get update` against it fails fast and we fall back to
# the live deb.debian.org mirror with an explicit `pkg=version` pin list
# (docker/chromium-deps-pinned.txt, captured via `dpkg -l` /
# /var/log/apt/history.log from a known-good build) so the layer is still
# reproducible even on that path.
ARG INSTALL_BROWSER=true
ARG DEBIAN_SNAPSHOT=20260824T000000Z
COPY docker/chromium-deps-pinned.txt ./docker/chromium-deps-pinned.txt
RUN if [ "$INSTALL_BROWSER" = "true" ]; then \
      pip install --no-cache-dir playwright==1.50.0 && \
      cp /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list.d/debian.sources.orig && \
      printf 'Types: deb\nURIs: https://snapshot.debian.org/archive/debian/%s\nSuites: trixie trixie-updates\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.pgp\nCheck-Valid-Until: no\n\nTypes: deb\nURIs: https://snapshot.debian.org/archive/debian-security/%s\nSuites: trixie-security\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.pgp\nCheck-Valid-Until: no\n' "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list.d/debian.sources && \
      if apt-get update; then \
        apt-get install -y --no-install-recommends \
          fonts-liberation libasound2 libatk-bridge2.0-0 libatk1.0-0 libatspi2.0-0 \
          libcairo2 libcups2 libdbus-1-3 libdrm2 libegl1 libgbm1 libglib2.0-0 \
          libgtk-3-0 libnspr4 libnss3 libpango-1.0-0 libx11-6 libx11-xcb1 libxcb1 \
          libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 libxshmfence1 \
          xvfb fonts-noto-color-emoji fonts-unifont libfontconfig1 libfreetype6 \
          xfonts-cyrillic xfonts-scalable fonts-ipafont-gothic fonts-wqy-zenhei \
          fonts-tlwg-loma-otf; \
      else \
        echo "WARNING: snapshot.debian.org unreachable at build time; falling" \
             "back to the live deb.debian.org mirror, pinned to exact" \
             "package=version pairs from docker/chromium-deps-pinned.txt for" \
             "reproducibility." >&2 && \
        mv /etc/apt/sources.list.d/debian.sources.orig /etc/apt/sources.list.d/debian.sources && \
        apt-get update && \
        grep -v '^#' docker/chromium-deps-pinned.txt | xargs apt-get install -y --no-install-recommends; \
      fi && \
      rm -f /etc/apt/sources.list.d/debian.sources.orig && \
      rm -rf /var/lib/apt/lists/* && \
      playwright install chromium && \
      chmod -R a+rX /ms-playwright; \
    fi

COPY pyproject.toml README.md ./
COPY tiblender_runtime ./tiblender_runtime
# The dev extra (pytest, coverage) so the image can run its own test suite --
# see the tests/ copy below and README.md "Tests and coverage". It adds a few
# MB to a ~1.9 GB image. The package itself goes in --no-deps because
# requirements.txt already pinned every runtime dependency above.
RUN pip install --no-cache-dir --no-deps . \
    && pip install --no-cache-dir ".[dev]"

COPY config ./config
COPY fixtures ./fixtures
COPY sample_runs ./sample_runs
COPY viewer ./viewer
COPY searxng ./searxng
COPY tools ./tools

# The two one-command scripts (the offline demo and the evaluation figures).
# They are small shell scripts; without them the Docker path cannot recompute
# the paper's figures.
COPY demo ./demo
COPY evaluation ./evaluation

# The pytest suite and the repository files some tests read, so the suite
# also runs on the Docker path.
COPY tests ./tests
COPY docs ./docs
COPY LICENSE CITATION.cff PIPELINE_OVERVIEW.md Makefile install.sh docker-compose.yml Dockerfile .dockerignore .env.example .gitignore ./

# Run as an unprivileged user whose uid/gid default to 1000 (the usual first
# host account). The container writes into bind-mounted ./output and ./viewer,
# so running as root would leave root-owned files the host user can neither
# clean nor overwrite -- which also broke a second `docker compose up` into
# the same output directory. Compose overrides `user:` with
# TIBLENDER_UID/TIBLENDER_GID for hosts whose uid is not 1000.
RUN groupadd --gid 1000 tiblender \
    && useradd --uid 1000 --gid 1000 --create-home --home-dir /home/tiblender tiblender \
    && mkdir -p /app/output /app/viewer \
    && touch /app/output/.gitkeep \
    && chown -R tiblender:tiblender /app
ENV HOME=/home/tiblender
USER tiblender

# ENTRYPOINT is the CLI itself, so a trailing argument is APPENDED to it:
# `docker compose run --rm tiblender doctor` runs `tiblender doctor`. With a
# bare CMD and no ENTRYPOINT it replaced the whole command instead and failed
# with `exec: "doctor": executable file not found`. CMD therefore carries only
# the default demo arguments; `tiblender demo` is run-cycles + verify-sample +
# build-viewer in one process.
#
# --report-json is part of the default arguments so that `docker compose up
# --build` writes `output/report.json`, exactly as `demo/run_demo.sh` does on
# the Python path, so both paths leave the same documented output.
#
# The ~95 MB dataset (`dataset/`) is deliberately NOT baked into the
# image: it would triple the layer that carries it for a directory the
# pipeline never reads. `docker-compose.yml` bind-mounts it read-only
# instead, so `evaluation/run_evaluation.sh` runs in the container too. See
# README.md, "Reproducing the Paper's Evaluation".
ENTRYPOINT ["tiblender"]
CMD ["demo", "--output-root", "output", "--cycles", "3", "--interval", "0", "--viewer-dir", "viewer", "--report-json", "output/report.json"]
