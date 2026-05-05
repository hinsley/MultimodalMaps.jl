# GCE/GCS runbook for attempt-055

This run uses the reusable image made from the earlier kneading cloud setup:
`attempt049-ready-20260429`.

## Target

- project: `codex-bigcomputations`
- preferred zone: `us-central1-a`
- suggested machine: `c4-standard-48` or equivalent 48-vCPU instance
- expected repo path on VM: `~/MultimodalMaps.jl`
- GCS artifact prefix: `gs://carter-kneading-attempt055/attempt-055`

The runner itself fixes `JULIA_NUM_THREADS=48` unless overridden.

## Start command on the VM

Run from `~/MultimodalMaps.jl`:

```bash
ATTEMPT055_GCS_URI=gs://carter-kneading-attempt055/attempt-055 \
  kneading/experiment/attempt-055/run_gce_grid500_tangent_zero.sh
```

The command is safe to run inside `tmux`; completed column checkpoint files are
resumable.

## Monitoring from local machine

Replace the instance name if a different one is used:

```bash
gcloud compute ssh --zone "us-central1-a" "attempt-055-48cpu" --project "codex-bigcomputations" --command '
cd ~/MultimodalMaps.jl
TAG=grid500_tangent_ca_dotzero_tmax1e5_iter8_ystub_sf_xfiltered
COL_DIR=kneading/experiment/attempt-055/${TAG}_columns
LOG=kneading/experiment/attempt-055/${TAG}.log
DONE=$(find "$COL_DIR" -maxdepth 1 -name "column_*.tsv" 2>/dev/null | wc -l | tr -d " ")
PCT=$(awk -v d="$DONE" "BEGIN { printf \"%.2f\", 100*d/500 }")
echo "$DONE/500 columns complete ($PCT%)"
tail -5 "$LOG" 2>/dev/null || true
tmux ls 2>/dev/null || true
'
```

## Retrieval

After completion:

```bash
mkdir -p kneading/experiment/attempt-055/gcs_results
gcloud storage cp --recursive \
  gs://carter-kneading-attempt055/attempt-055/* \
  kneading/experiment/attempt-055/gcs_results/
```

Expected key outputs:

- `grid500_tangent_ca_dotzero_tmax1e5_iter8_ystub_sf_xfiltered_results.tsv`
- `grid500_tangent_ca_dotzero_tmax1e5_iter8_ystub_sf_xfiltered_contours.png`
- `grid500_tangent_ca_dotzero_tmax1e5_iter8_ystub_sf_xfiltered_summary.txt`
- `grid500_tangent_ca_dotzero_tmax1e5_iter8_ystub_sf_xfiltered.log`

## Cleanup

Stop the VM after upload is verified to stop compute charges. Delete it if the
boot disk is not needed.
