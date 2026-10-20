# TIBlender Report Viewer

Static browser viewer for generated TIBlender reports. It is a **variant** of
the dataset viewer (`../dataset/viewer/`), not a verbatim copy:

- `styles.css` is byte-identical to the dataset viewer's.
- `index.html` and `app.js` differ. The dataset viewer browses the 8,288
  released reports and adds two controls this one does not have — a
  **platform filter** and a **report-id lookup box** — because its payload
  carries a per-report source platform and stable released report ids. A
  run of this code produces `TIBLENDER-RUN-*` ids for whatever cycles you
  ran, so those two controls have nothing to filter on here. Everything
  else — the severity and threat-category filters, pagination, and the
  detail panel — is the same, and this copy's `index.html`/`app.js` are
  formatted (expanded) rather than minified into long lines.

The only file this directory does not ship is the generated data file,
`reports_viewer_data.js`, which the CLI builds from your own report output.

## Build the data file

From the repository root (`tiblender_runtime`'s parent directory), after
running one or more pipeline cycles into an output directory (a single
`output/reports/*.json` tree, or a multi-cycle `output/cycle_00N/reports/*.json`
tree both work):

```bash
tiblender build-viewer --reports-root output --viewer-dir viewer
```

This writes `viewer/reports_viewer_data.js` (a `window.TIBLENDER_ARTIFACT_DATA`
assignment) and prints the data-file path, the number of reports bundled, and
the number of report files skipped (unreadable or not valid JSON, e.g. a
report from a cycle still in progress — these are logged as warnings, not
treated as a build failure):

```
viewer_data=viewer/reports_viewer_data.js
reports=19
skipped=0
```

`reports=` counts whatever report JSON is under `--reports-root`, so it
grows as you accumulate cycles; `19` is what the bundled offline
three-cycle demo produces.

Note: the payload's top-level `generated_at` field is a **build** timestamp —
when this data file was assembled — not a report timestamp. Under a fixed
clock (`TIBLENDER_MODE=offline`, or `TIBLENDER_CLOCK=fixed`) it is taken from
the newest report in the bundle rather than from wall time, so rebuilding the
same reports rewrites exactly the same bytes and two consecutive offline runs
are byte-identical here too. On a wall-clock run (`auto`/`live`) it is the
time of the build and does change on every rebuild.

## Serve it

```bash
python3 -m http.server 8765 --directory viewer
```

Then open `http://localhost:8765/` in a browser. Re-run `build-viewer` after
generating more reports and refresh the page to pick up the update.

`make viewer` does the same thing from the repository root, and
`docker compose --profile viewer up viewer` does it in a container. Both
default to port 8765; override with `make viewer VIEWER_PORT=8899` or
`TIBLENDER_VIEWER_PORT=8899 docker compose --profile viewer up viewer` if that
port is already in use on the machine.
