# TIBlender

**Early-Warning Threat Intelligence from Cross-Platform Social Media Evidence**

This repository contains the implementation and the released dataset of the
ACSAC 2026 paper:

> Hiroki Nakano, Takashi Koide, and Daiki Chiba. *TIBlender: Early-Warning
> Threat Intelligence from Cross-Platform Social Media Evidence.* In
> Proceedings of the 42nd Annual Computer Security Applications Conference
> (ACSAC 2026), 2026.

Project page: <https://tiblender.github.io/>

## Overview

TIBlender collects threat-related posts from X, Reddit, Telegram, and
Discord, groups them into campaign-level clusters, and investigates each
cluster with multi-perspective, RL-guided LLM agents and external lookup
tools. Two LLM judges evaluate the findings, and TIBlender writes structured
threat intelligence reports (JSON, Markdown, PDF, STIX 2.1) with verified
IoCs and evidence-backed justifications. Two feedback loops carry state
across cycles: trend-driven dynamic query generation and RL policy training.

The runtime has three modes:

- **offline**: deterministic, with no credentials and no network access.
  Anonymized posts from the released dataset stand in for the collectors,
  recorded tool responses stand in for the external tools, and a
  deterministic heuristic stands in for the LLM.
- **live**: real platforms and a real LLM provider, with your own
  credentials.
- **auto** (the CLI default outside Docker Compose): live where credentials
  are configured, offline otherwise, decided per service.

The offline demo shows that the whole pipeline runs end to end and
reproducibly. Its report content comes from the heuristic, not an LLM. The
paper's evaluation results come from the 31-day deployment and are
recomputed from the released dataset in `dataset/`.

## Repository Layout

| Path | Contents |
|---|---|
| `tiblender_runtime/` | The pipeline: collection, clustering, investigation, evaluation, reporting, and both feedback loops |
| `demo/` | `run_demo.sh` (offline demo) and `run_dqn_demo.sh` (optional DQN fine-tuning demo) |
| `evaluation/` | `run_evaluation.sh`, which recomputes the paper's evaluation figures from `dataset/` |
| `dataset/` | The released dataset: 8,288 reports, evaluation evidence, the analysis notebook, and a browser viewer (see [`dataset/README.md`](dataset/README.md)) |
| `fixtures/` | Anonymized collector inputs and recorded tool responses for offline mode |
| `sample_runs/` | The expected output the offline demo is verified against, and a small sample input |
| `config/` | Keyword catalog and collection seed lists |
| `viewer/` | Static browser viewer for generated reports |
| `docs/` | [`REFERENCE.md`](docs/REFERENCE.md) (CLI, configuration, output layout), [`FIDELITY.md`](docs/FIDELITY.md) (differences from the deployed system), [`CREDENTIALS.md`](docs/CREDENTIALS.md) (live-mode setup), [`ethics.md`](docs/ethics.md) |
| `PIPELINE_OVERVIEW.md` | One cycle, stage by stage |
| `tests/` | pytest suite |
| `tools/` | Utilities used to build the fixtures and compare evaluation figures |

## Requirements

- Linux on x86_64 (tested on Ubuntu 24.04), 2–4 CPU cores, 4–8 GiB RAM,
  and about 8 GiB of free disk. No GPU is needed.
- Either Docker Engine with Docker Compose v2, or Python 3.11+ with `make`
  and `git`. On Ubuntu 22.04, which ships Python 3.10, install
  `python3.11 python3.11-venv` and pass `--python python3.11` to
  `install.sh`.
- Network access only to build the image or install the Python packages.
- No credentials for the offline demo, the evaluation script, or the tests.

## Quick Start

With Docker:

```bash
docker compose up --build
```

This builds the image and runs three offline cycles, verifies them against
the expected output, and builds the viewer data. If your uid is not 1000,
prefix the command with `TIBLENDER_UID=$(id -u) TIBLENDER_GID=$(id -g)`.

With Python:

```bash
bash install.sh --offline --with-dev     # create .venv with pinned dependencies
bash demo/run_demo.sh                    # three offline cycles + verification
make viewer                              # browse the reports at http://localhost:8765/
```

`demo/run_demo.sh` creates `.venv` itself if it does not exist, and it runs
from any working directory. `make clean-output` removes the generated
output. To run the CLI by hand, set `TIBLENDER_MODE=offline`; otherwise the
CLI defaults to `auto` mode, which uses live services where it can.

## Expected Output

The offline demo prints these counters (only timestamps vary between runs):

```
cycles_run=3
cycle_001_posts=28
cycle_001_reports=3
cycle_001_dynamic_queries=5
cycle_001_tool_recording_hit_rate=1.000
cycle_002_posts=33
cycle_002_reports=10
cycle_002_dynamic_queries=2
cycle_002_tool_recording_hit_rate=1.000
cycle_003_posts=45
cycle_003_reports=6
cycle_003_dynamic_queries=3
cycle_003_tool_recording_hit_rate=0.959
verify_sample: ok (0 diffs)
viewer_data=viewer/reports_viewer_data.js
reports=19
skipped=0
report_json=output/report.json
```

`verify_sample: ok (0 diffs)` means the run matches the committed
expectation in `sample_runs/offline_three_cycles/expected/` in every compared
field: post, cluster, and report counts, report ids and IoCs, dynamic
queries, trend counts, and RL training status. The 19 reports are written
under `output/cycle_00N/` as JSON with Markdown, PDF, and STIX sidecars, and
all of them are also collected in `output/report.json`. The output layout is
described in [`docs/REFERENCE.md`](docs/REFERENCE.md#output-layout).

## Reproducing the Paper's Evaluation

```bash
bash evaluation/run_evaluation.sh
```

This executes a copy of `dataset/notebook/evaluation_figures.ipynb` on the
released records and prints `evaluation: ok` when the recomputed figures are
byte-identical to `dataset/notebook/evaluation_figures.json`. It covers the
evaluation tables and summary metrics of Sections IV-B to IV-G of the paper.
[`evaluation/README.md`](evaluation/README.md) maps the figures to the paper
sections, and [`dataset/README.md`](dataset/README.md) lists the expected
values. On Docker, run it as
`docker compose run --rm --entrypoint bash tiblender evaluation/run_evaluation.sh`.

## Optional: DQN Fine-Tuning Demo

The three bundled cycles collect fewer replay transitions than the paper's
1,000-transition training threshold, so the demo above runs behaviour
cloning only. This script reruns the same inputs with the threshold lowered
to 100, so cycles 2 and 3 also run DQN fine-tuning:

```bash
bash demo/run_dqn_demo.sh                # or: make dqn-demo
```

It prints per-cycle training status and DQN loss, writes
`output/dqn_demo/dqn_training.json`, and prints `dqn_demo: ok` on success. It shows
that the DQN stage runs and updates the policy, not that the policy
improves; a few hundred transitions are far too few for that.

## Live Mode

Live mode needs your own credentials: an X API Basic bearer token, a Reddit
script app, a Telegram `api_id`/`api_hash` and user session, a Discord
user-account or bot token, a VirusTotal key, and one OpenAI-compatible or
Azure LLM endpoint. [`docs/CREDENTIALS.md`](docs/CREDENTIALS.md) explains
each one.

```bash
cp .env.example .env    # fill in one LLM provider and your credentials
docker compose --profile live up --build
```

`tiblender doctor` reports which services resolved to live, fixture,
heuristic, or unavailable. Live runs are not deterministic.

## Tests

```bash
make test                                              # Python
docker compose run --rm --entrypoint pytest tiblender -q   # Docker
```

## Limitations

[`docs/FIDELITY.md`](docs/FIDELITY.md) lists every difference from the
deployed system. The main ones:

- Any OpenAI-compatible or Azure endpoint can serve as the LLM; the paper
  used Grok 4 Fast. In offline mode, a deterministic heuristic replaces the
  LLM, so offline reports can include off-topic posts or irrelevant
  indicators.
- The default embeddings are deterministic hashing vectors; the paper's
  models (bge-large-en-v1.5, all-MiniLM-L6-v2) are available with
  `install.sh --with-reference`.
- With only three bundled cycles, the RL policy stays below its training
  threshold, and only the `EARLY_WARNING` trend axis fires.
- Discord and Telegram collection reads only communities the configured
  account has already joined. Discovering and joining communities is not
  part of this repository.
- The production databases, API server, frontend, and worker pool of the
  deployment are not included.

## Ethics

The posts in `dataset/` and `fixtures/` are excerpts of publicly posted
content. Author identifiers and other identifying metadata of each post are
removed. The post text in `dataset/` is kept as posted, so it can mention
other accounts, email addresses (often phishing senders), and links; the
fixture text is anonymized further. Do not use this repository to
re-identify posters or the people mentioned in posts.

The repository is not uniformly defanged. Recorded tool responses,
generated reports, and part of the post text contain real, possibly
malicious indicators in plain form: do not open them. These IoC lists are
research data, not a curated blocklist. See
[`docs/ethics.md`](docs/ethics.md) for the full statement.

## License

The code is released under the TIBlender Research and Evaluation License
([`LICENSE`](LICENSE)). Any individual or organization, including the
research divisions of companies, may use, modify, and share it for
non-commercial research, including reproducing the paper's results and
building research artifacts on it. Shared copies and modified versions must
stay under the same license, and no patent rights are granted. The dataset
in `dataset/` is released under CC BY-NC 4.0
([`dataset/LICENSE`](dataset/LICENSE)). For any other use, including
commercial use, contact the authors through <https://tiblender.github.io/>.

## Citation

```bibtex
@inproceedings{nakano2026tiblender,
  title     = {TIBlender: Early-Warning Threat Intelligence from Cross-Platform Social Media Evidence},
  author    = {Hiroki Nakano and Takashi Koide and Daiki Chiba},
  booktitle = {Proceedings of the 42nd Annual Computer Security Applications Conference (ACSAC 2026)},
  year      = {2026}
}
```
