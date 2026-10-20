# Recomputing the paper's evaluation results

This directory recomputes the evaluation tables and summary metrics of the
paper's Sections IV-B to IV-G from the released records, rather than reading them
back from the paper.

Nothing here reimplements the evaluation. The computation lives in the dataset's
notebook, `dataset/notebook/evaluation_figures.ipynb`; this
directory is the one-command way to execute it and check the result.

## One command

```bash
bash evaluation/run_evaluation.sh
```

It prints `evaluation: ok` and exits 0 when the recomputed figures match the
released `dataset/notebook/evaluation_figures.json`; otherwise it prints the
keys that differ and exits 1. Runnable from any working directory. No
credentials and no external services; network access is needed only if the
Jupyter dependencies still have to be installed.

What it does, in order:

1. resolves the dataset — `dataset/` in this repository; `DATASET_DIR`
   overrides it;
2. ensures a Python ≥3.11 environment with `jupyter`/`nbclient`, creating
   `.venv` with `install.sh --offline --with-dev` if the current interpreter
   has neither;
3. copies the notebook to `output/evaluation/notebook/` and links
   `output/evaluation/raw_dataset` to `dataset/raw_dataset`, because the
   notebook resolves its inputs as `raw_dataset` one level above its own
   directory. **The dataset is never written to**;
4. executes the copy with `jupyter nbconvert --to notebook --execute`, which
   also leaves an executed copy at
   `output/evaluation/notebook/evaluation_figures_executed.ipynb`;
5. compares the `evaluation_figures.json` the run produced with the released
   one, byte for byte (`tools/compare_evaluation_figures.py`).

Use `--output-root DIR` to put the working copy somewhere else.

First run: **10-60 s**, most of it the one-off
`jupyter` install; the notebook itself executes in a few seconds once the
dependency is present.

## On the Docker path

The image carries this directory but not the ~95 MB dataset;
`docker-compose.yml` bind-mounts it read-only at `/app/dataset`
instead. The image's entry point is the `tiblender` CLI, so reach the script
through a shell:

```bash
docker compose run --rm --entrypoint bash tiblender evaluation/run_evaluation.sh
```

## Running the notebook by hand

```bash
pip install jupyter                # or: bash install.sh --offline --with-dev
cd dataset/notebook
jupyter nbconvert --to notebook --execute evaluation_figures.ipynb \
  --output evaluation_figures_executed.ipynb
```

Executing it in place rewrites `dataset/notebook/evaluation_figures.json`, so
prefer the script if you want the released copy left untouched. The dataset
also ships pre-executed copies (`evaluation_figures_executed.ipynb`
and `.html`) for reading without running anything.

## What each figure group recomputes

| Notebook section | Paper | Recomputed from |
|---|---|---|
| 1. Operational overview | §IV-B (31-day deployment: raw posts per platform, clusters, Step 2 and Step 4 filtering, runtime) | `raw_dataset/evaluation_evidence/pipeline_overview.json`, cross-checked against the report count in `raw_dataset/reports.jsonl` |
| 2. Categories, severity, cross-platform reports, IoCs per report | §IV-B, §IV-C | `raw_dataset/reports.jsonl` |
| 3. Platform contribution and single-platform removal loss | §IV-C (cross-platform monitoring) | `raw_dataset/reports.jsonl` |
| 4. Same-input baseline comparison — totals, per-category counts, novelty, coverage, VirusTotal detection distribution, Tranco match | §IV-D (baseline comparison) | `same_input_direct_records.jsonl`, `baseline_ioc_records.jsonl` |
| 5. Public-feed denominators, overlap and feed-scoped absence | §IV-E (overlap with public threat feeds) | `public_feed_overlap_records.jsonl` |
| 6. Early-detection rate and mean lead time | §IV-E (lead time) | `public_feed_timing_records.jsonl` |
| 7. Ablation membership counts and yield parameters | §IV-G (ablation) | `ablation_membership.jsonl`, `ablation_parameters.json` |
| 8. Manual quality audit and false-negative review | §IV-F (human validation) | `manual_review_records.jsonl`, `manual_review_patterns.json`, `skip_cluster_units.jsonl` |
| 9. Save | — | writes `evaluation_figures.json` |

## Reading the numbers

The expected values, the **sign convention for lead time** (negative means
TIBlender was first), the KEV reconciliation and the per-file record schemas
are all in `dataset/README.md`. They are not repeated here.

Two limits `dataset/README.md` states and this script does not change: OTX
campaign-level pairing is not computed, and the ablation parameters are stored
aggregate values rather than being reconstructed from per-investigation
evidence.

## If it fails

- **`error: no dataset at ...`**, naming the directory it looked in — the
  dataset is not at `dataset/` in this repository.
  Unpack it as `dataset/`, or run `DATASET_DIR=/path/to/dataset
  evaluation/run_evaluation.sh`.
- **`evaluation: FAILED`** reporting a count of differing values — the script names each
  differing key path. A difference means the released records and the released
  figures disagree; report it, do not edit either file.
- **`evaluation: FAILED`** reporting `the same values ... but not the same
  bytes: formatting or key order differs` — the values match and only
  the formatting differs (key order, indentation, trailing newline), usually a
  different `json` writer. The claim is byte equality, so this is still a
  failure, but not a change in any result.
