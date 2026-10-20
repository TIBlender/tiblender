.PHONY: install offline live test coverage doctor show-config audit-contract \
        viewer docker-offline docker-live record-tool-responses build-fixtures \
        harvest-tool-queries dqn-demo \
        clean clean-output

# Ubuntu 22.04/24.04 ship `python3` only (there is no `python` unless
# python-is-python3 is installed), and install.sh already defaults to python3.
PYTHON ?= python3
VENV ?= .venv
VENV_BIN := $(VENV)/bin
TIBLENDER := $(VENV_BIN)/tiblender
PYTHONPATH := .
OUTPUT_ROOT ?= output
VIEWER_DIR ?= viewer
VIEWER_PORT ?= 8765

# Install into a local virtualenv (see install.sh for the flags this wraps).
# --with-dev pulls in pytest/coverage so `make test`/`make coverage` work
# out of the documented install -> make workflow without an extra step.
install:
	bash install.sh --python $(PYTHON) --with-dev

# Deterministic offline three-cycle demo: no credentials required. Pin the
# same TIBLENDER_MODE/PYTHONHASHSEED/TIBLENDER_CLOCK the Dockerfile sets as
# ENV defaults -- the runtime's own default is TIBLENDER_MODE=auto, which on
# a networked host makes live tool calls and produces a verify-sample diff
# instead of the deterministic fixture-backed run this target promises.
offline:
	TIBLENDER_MODE=offline PYTHONHASHSEED=0 TIBLENDER_CLOCK=auto $(TIBLENDER) run-cycles --output-root $(OUTPUT_ROOT) --cycles 3 --interval 0
	TIBLENDER_MODE=offline PYTHONHASHSEED=0 TIBLENDER_CLOCK=auto $(TIBLENDER) verify-sample --output-root $(OUTPUT_ROOT)
	TIBLENDER_MODE=offline PYTHONHASHSEED=0 TIBLENDER_CLOCK=auto $(TIBLENDER) build-viewer --reports-root $(OUTPUT_ROOT) --viewer-dir $(VIEWER_DIR)

# Optional: the same fixtures with the RL training threshold lowered to 100,
# so the DQN fine-tuning stage runs (see demo/run_dqn_demo.sh). Writes only
# under $(OUTPUT_ROOT)/dqn_demo.
dqn-demo:
	TIBLENDER_MODE=offline PYTHONHASHSEED=0 TIBLENDER_CLOCK=auto PYTHONPATH=$(PYTHONPATH) $(TIBLENDER) dqn-demo --output-root $(OUTPUT_ROOT)/dqn_demo

# One live cycle against real collectors/tools/LLM. Requires credentials in
# .env (see .env.example); fails fast if none are configured.
live:
	TIBLENDER_MODE=live $(TIBLENDER) run-cycles --output-root $(OUTPUT_ROOT) --cycles 1

# Self-heal: pytest only exists once the dev extra is installed (`make
# install` does this by default; a pre-existing venv from an older install
# may not have it yet). Install it on demand rather than failing outright.
test:
	@if [ ! -x "$(VENV_BIN)/pytest" ]; then \
		echo "==> $(VENV_BIN)/pytest not found; installing the dev extra"; \
		$(VENV_BIN)/pip install --no-cache-dir ".[dev]"; \
	fi
	TIBLENDER_MODE=offline PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) $(VENV_BIN)/pytest -q

coverage:
	@if [ ! -x "$(VENV_BIN)/pytest" ]; then \
		echo "==> $(VENV_BIN)/pytest not found; installing the dev extra"; \
		$(VENV_BIN)/pip install --no-cache-dir ".[dev]"; \
	fi
	TIBLENDER_MODE=offline PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) $(VENV_BIN)/coverage run -m pytest -q
	$(VENV_BIN)/coverage report -m

doctor:
	PYTHONPATH=$(PYTHONPATH) $(TIBLENDER) doctor

show-config:
	PYTHONPATH=$(PYTHONPATH) $(TIBLENDER) show-config

audit-contract:
	PYTHONPATH=$(PYTHONPATH) $(TIBLENDER) audit-contract

# Serve the built report viewer locally (run `make offline` first).
viewer:
	$(PYTHON) -m http.server $(VIEWER_PORT) --directory $(VIEWER_DIR)

# Build and run the offline demo in Docker (default compose profile).
docker-offline:
	docker compose up --build

# Also start the bundled SearXNG instance for live web search.
docker-live:
	docker compose --profile live up --build

# Maintainer-only: record tool responses from live providers into fixtures/tools.
# Requires network access and any live tool credentials in .env.
QUERIES ?= fixtures/tools/queries.json
record-tool-responses:
	$(VENV_BIN)/python tools/record_tool_responses.py --fixtures fixtures --queries $(QUERIES)

# Maintainer-only: rebuild the anonymized collector fixtures from the released
# dataset. DATASET_PATH defaults to `dataset/` in this repository; set it to
# point anywhere else.
DATASET_PATH ?= dataset/raw_dataset/reports.jsonl
build-fixtures:
	$(VENV_BIN)/python tools/build_fixtures.py \
		--dataset $(DATASET_PATH) --out fixtures/collectors --cycles 3 --seed 20260909

# Maintainer-only, no credentials, no network: run the deterministic offline demo,
# collect every external-tool request key that found no recorded response,
# merge it into fixtures/tools/harvested_queries.json, and rebuild
# queries.json. Prints the per-tool replay hit-rate table. Follow with
# `make record-tool-responses` (which needs network + credentials) and repeat
# until this target reports zero misses. See tools/harvest_tool_queries.py.
harvest-tool-queries:
	$(VENV_BIN)/python tools/harvest_tool_queries.py \
		--fixtures fixtures --dataset $(DATASET_PATH) --cycles 3 --seed 20260909

# Remove only generated run output (not the venv), so the next
# `make offline` / `docker compose up` starts from a clean tree. The
# container runs as a non-root user with the host's uid (see
# docker-compose.yml), so these files are owned by the invoking user and a
# plain rm is enough -- no privileged helper container required. The empty
# output directory is recreated afterwards because Docker Compose
# bind-mounts it: dockerd creates a missing host directory as root, and the
# container's unprivileged user could then not write into it. The .gitkeep
# sentinel inside it is TRACKED (see .gitignore), so it is restored too --
# otherwise `make clean-output` leaves the worktree dirty.
clean-output:
	rm -rf $(OUTPUT_ROOT) $(VIEWER_DIR)/reports_viewer_data.js
	mkdir -p $(OUTPUT_ROOT)
	touch $(OUTPUT_ROOT)/.gitkeep

clean:
	rm -rf $(OUTPUT_ROOT) $(VENV) build *.egg-info
