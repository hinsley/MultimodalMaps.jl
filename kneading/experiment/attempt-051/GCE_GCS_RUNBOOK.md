# GCE/GCS runbook for attempt-051

This attempt is designed for the existing Google Compute Engine image and
bucket workflow used in attempts 048-050.

## VM target

- project: `codex-bigcomputations`
- zone: `us-central1-a`
- instance: `instance-20260429-083718`
- expected repo path on VM: `~/MultimodalMaps.jl`
- expected Julia path: `~/.juliaup/bin/julia`
- runner Julia choice: uses `julia +release` when that channel exists, otherwise
  falls back to plain `julia`
- GCS artifact prefix: `gs://carter-kneading-attempt048/attempt-051`

SSH:

```bash
gcloud compute ssh --zone "us-central1-a" "instance-20260429-083718" --project "codex-bigcomputations"
```

## Start command

Run inside a `tmux` session from `~/MultimodalMaps.jl`:

```bash
ATTEMPT051_GCS_URI=gs://carter-kneading-attempt048/attempt-051 \
  kneading/experiment/attempt-051/run_grid1000_seq12_tmax1e5_gh_sweep.sh
```

The runner performs the three `g_h` cases serially:

- `g_h = 0.0` with label `gh0p000`
- `g_h = 1.0e-3` with label `gh0p001`
- `g_h = 1.0e-2` with label `gh0p01`

For each case it computes columns, writes a merged TSV, generates prefix plots,
generates the filtered final contour plot, and uploads final artifacts to GCS.
If interrupted, completed column files remain resumable.

## Monitoring

Use this local command to see concise progress and the last five log lines for
each case:

```bash
gcloud compute ssh --zone "us-central1-a" "instance-20260429-083718" --project "codex-bigcomputations" --command '
cd ~/MultimodalMaps.jl
for LABEL in gh0p000 gh0p001 gh0p01; do
  TAG=grid1000_seq12_tmax1e5_${LABEL}_prefixes_remap40_newmodel
  COL_DIR=kneading/experiment/attempt-051/${TAG}_columns
  LOG=kneading/experiment/attempt-051/${TAG}.log
  DONE=$(find "$COL_DIR" -maxdepth 1 -name "column_*.tsv" 2>/dev/null | wc -l | tr -d " ")
  PCT=$(awk -v d="$DONE" "BEGIN { printf \"%.2f\", 100*d/1000 }")
  echo "$LABEL: $DONE/1000 columns complete ($PCT%)"
  tail -5 "$LOG" 2>/dev/null || true
  echo
done
tmux ls 2>/dev/null || true
'
```

## Retrieval

After completion, retrieve all uploaded artifacts locally:

```bash
mkdir -p kneading/experiment/attempt-051/gcs_results
gcloud storage cp --recursive \
  gs://carter-kneading-attempt048/attempt-051/* \
  kneading/experiment/attempt-051/gcs_results/
```

Expected important files for each label:

- `grid1000_seq12_tmax1e5_<label>_prefixes_remap40_newmodel_results.tsv`
- `grid1000_seq12_tmax1e5_<label>_prefixes_remap40_newmodel_contours.png`
- `grid1000_seq12_tmax1e5_<label>_prefixes_remap40_newmodel_prefixNN_contours.png`
- `grid1000_seq12_tmax1e5_<label>_prefixcompatible_tzero2to12_contours.png`
- `grid1000_seq12_tmax1e5_<label>_prefixcompatible_tzero2to12_summary.txt`

The upload intentionally includes TSVs, logs, legends, prefix plots, and final
filtered plots, but not a VM image or boot disk artifact.
