# GCE/GCS runbook for attempt-049

This attempt is ready to run on Google Cloud, but the compute service is
**Google Compute Engine (GCE)**, not Google Cloud Storage (GCS).

- Use **GCE** to rent a C3 or N2 VM and run the Julia scan.
- Use **GCS** only if you want a cheap remote backup location for logs, column
  checkpoint TSVs, merged TSVs, and PNG outputs.

The scan is column-checkpointed. If a Spot VM is preempted, restart the same
script in the same checkout/disk and completed columns are skipped.

## Recommended VM

Start with a Spot C4/C3 VM in `us-central1` if quota/capacity allow it.
The previous successful run used a 24-vCPU VM after quota reductions, which is
good enough; larger machines should mainly reduce wall-clock time.

Fallbacks:

- `c3-standard-44` if 88 vCPUs are unavailable
- `n2-standard-48` if C3 Spot capacity is unavailable
- on-demand C3/N2 if interruptions are more annoying than the extra cost

Use a persistent boot disk or attached persistent disk. Do not rely on local
SSD for the only copy of results.

## VM setup

On a fresh Debian/Ubuntu VM:

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates build-essential tmux
curl -fsSL https://install.julialang.org | sh -s -- -y
source ~/.bashrc
git clone https://github.com/hinsley/MultimodalMaps.jl.git
cd MultimodalMaps.jl
```

Instantiate once before the run. The attempt runner also does this by default,
but running it explicitly makes dependency failures happen before the long job.

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Lessons from the first GCE setup

The first live setup exposed a few issues that should be avoided next time:

- A default `10 GB` boot disk is too small. Julia packages/artifacts plus this
  repository filled the disk during setup. Start with at least `50 GB`.
- Pick VM access scopes that allow writing to GCS. `devstorage.read_only` lets
  the VM list/read the bucket but prevents final artifact upload.
- Give the VM service account object-write permission on the bucket, for
  example `roles/storage.objectAdmin` on the attempt bucket.
- If VM scopes are changed after boot, clear cached gcloud tokens on the VM or
  restart with a fresh token. A stale token can keep reporting
  `Provided scope(s) are not authorized` even after the VM scope is fixed.
- Run one tiny GCS upload test before starting the expensive scan.
- After setup is complete and before starting the scan, create a custom image
  from the prepared boot disk. That avoids paying again for Julia install,
  package download, and precompile time on future fresh VMs.
- Detached `tmux` commands may not load the same shell startup files as an
  interactive SSH session. The runner now prepends `~/.juliaup/bin` to `PATH`
  when present so Julia is found without manually exporting `PATH`.
- Minimal Debian images may not have `/usr/bin/time`. The runner now uses it
  when available and falls back to shell `time` otherwise.
- Headless VMs can report GLMakie/OpenGL and
  `DynamicalSystemsVisualizations` precompile failures during
  `Pkg.instantiate()`. For this attempt those were nonfatal because the scan
  uses CairoMakie and then successfully entered `contours.jl`.

Concrete commands from the first setup:

```bash
# Resize an accidentally small boot disk from the local machine.
gcloud compute disks resize VM_DISK_NAME \
  --zone=us-central1-c \
  --project=codex-bigcomputations \
  --size=50GB \
  --quiet

# Grow the Linux partition/filesystem on the VM.
sudo apt-get update
sudo apt-get install -y cloud-guest-utils
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
df -h /

# Allow the default VM service account to write attempt artifacts.
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET_NAME \
  --member=serviceAccount:VM_SERVICE_ACCOUNT_EMAIL \
  --role=roles/storage.objectAdmin

# If scopes were changed after the VM was already running, clear cached tokens
# before retesting GCS upload.
rm -f ~/.config/gcloud/access_tokens.db ~/.config/gcloud/credentials.db

# Must succeed before launching the scan.
date -Is > /tmp/attempt049_gcs_test.txt
gcloud storage cp /tmp/attempt049_gcs_test.txt \
  gs://YOUR_BUCKET_NAME/attempt-049/setup_test.txt
```

If a first launch fails before any columns are written, inspect the appended
log and restart after fixing the setup issue:

```bash
tail -220 kneading/experiment/attempt-049/grid1000_seq12_prefixes_remap40_newmodel.log
tmux kill-session -t attempt049 2>/dev/null || true
tmux new -d -s attempt049 \
  'ATTEMPT049_GCS_URI=gs://YOUR_BUCKET_NAME/attempt-049 kneading/experiment/attempt-049/run_grid1000_seq12_prefixes_remap40_newmodel.sh'
```

The log is append-only by design, so old failure text can remain above the
current run. Identify the current run by the newest timestamped
`Running attempt-049 full scan` block.

For the first VM, the service account was:

```text
82681361968-compute@developer.gserviceaccount.com
```

and the intended working bucket prefix for this attempt is:

```text
gs://carter-kneading-attempt048/attempt-049
```

## Prepared image workflow

Once a VM has Julia, the repository, instantiated packages, a large enough disk,
and verified GCS upload access, save it as a custom image before starting the
long scan:

```bash
gcloud compute instances stop VM_NAME \
  --zone=us-central1-c \
  --project=codex-bigcomputations

gcloud compute images create attempt049-ready-image \
  --source-disk VM_DISK_NAME \
  --source-disk-zone=us-central1-c \
  --project=codex-bigcomputations
```

Future VMs can be created from `attempt049-ready-image` and should be ready to
run without repeating dependency setup. Still verify GCS upload on the new VM,
because service-account scopes/IAM are VM/project settings, not just files on
disk.

## Run

Use `tmux` so the job survives SSH disconnects:

```bash
tmux new -s attempt049
cd ~/MultimodalMaps.jl
kneading/experiment/attempt-049/run_grid1000_seq12_prefixes_remap40_newmodel.sh
```

The runner defaults `JULIA_NUM_THREADS` to all visible CPUs via `nproc`, sets
`OPENBLAS_NUM_THREADS=1`, appends to the log instead of overwriting it, and
flushes filesystem buffers on shutdown signals.

To override core count:

```bash
JULIA_NUM_THREADS=44 kneading/experiment/attempt-049/run_grid1000_seq12_prefixes_remap40_newmodel.sh
```

To resume after preemption or manual stop, run the same command again. The
checkpoint directory is:

```text
kneading/experiment/attempt-049/grid1000_seq12_prefixes_remap40_newmodel_columns/
```

## GCS-backed run

If you want the VM to automatically upload both the resumable checkpoint data
and the generated kneading diagram figures, set `ATTEMPT049_GCS_URI` when you
start the runner:

```bash
ATTEMPT049_GCS_URI=gs://YOUR_BUCKET_NAME/attempt-049 \
  kneading/experiment/attempt-049/run_grid1000_seq12_prefixes_remap40_newmodel.sh
```

With `ATTEMPT049_GCS_URI` set, the runner does three things:

- syncs the column checkpoint directory on shutdown or failure
- uploads the log on shutdown or failure
- after a successful run, uploads the merged TSVs, legends, and all generated
  kneading diagram PNGs matching `grid1000_seq12_prefixes_remap40_newmodel*`

This means a complete run on the VM can generate the figures and leave both
the raw SSCS data and the PNGs retrievable from GCS after the VM is stopped or
deleted.

## Optional manual GCS backup

Create a bucket in the same region as the VM, for example `us-central1`, to
avoid unnecessary cross-region transfer:

```bash
gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=us-central1
```

During a long run, periodically copy checkpointed columns and logs:

```bash
gcloud storage rsync -r \
  kneading/experiment/attempt-049/grid1000_seq12_prefixes_remap40_newmodel_columns \
  gs://YOUR_BUCKET_NAME/attempt-049/grid1000_seq12_prefixes_remap40_newmodel_columns

gcloud storage cp \
  kneading/experiment/attempt-049/grid1000_seq12_prefixes_remap40_newmodel.log \
  gs://YOUR_BUCKET_NAME/attempt-049/
```

After completion, copy the merged data and plots:

```bash
gcloud storage cp kneading/experiment/attempt-049/grid1000_seq12_prefixes_remap40_newmodel* \
  gs://YOUR_BUCKET_NAME/attempt-049/
```

The TSV/PNG/log artifacts are small enough that GCS cost should be negligible
for this use. The main cost is GCE VM uptime.

## Retrieve results

If you used the GCS-backed runner or manual GCS backup, download the completed
data and figures from your local machine with:

```bash
mkdir -p attempt-049-results
gcloud storage cp -r \
  gs://YOUR_BUCKET_NAME/attempt-049/grid1000_seq12_prefixes_remap40_newmodel* \
  attempt-049-results/
```

If you did not use GCS, copy directly from the VM before deleting it:

```bash
mkdir -p attempt-049-results
gcloud compute scp --recurse \
  VM_NAME:~/MultimodalMaps.jl/kneading/experiment/attempt-049/grid1000_seq12_prefixes_remap40_newmodel* \
  attempt-049-results/ \
  --zone=YOUR_ZONE
```

If the VM was preempted but the persistent disk remains, recreate or restart a
VM with that disk attached, then use the direct `gcloud compute scp` command
above or upload to GCS from the recovered VM.

## Rebuild figures from saved data

If the sweep data are complete but the figure-generation step was interrupted,
rerun the same command on the VM. Completed columns are skipped, then the
script rebuilds the merged TSV, legends, and all prefix contour figures from
the checkpointed columns before uploading final artifacts to GCS.

If the completed results TSV is local but you only want to replay plots without
resolving ODEs again, use:

```bash
ATTEMPT049_PLOT_ONLY_RESULTS=kneading/experiment/attempt-049/grid1000_seq12_prefixes_remap40_newmodel_results.tsv \
ATTEMPT049_OUTPUT_TAG=grid1000_seq12_prefixes_remap40_newmodel_replot \
ATTEMPT049_GCS_URI=gs://YOUR_BUCKET_NAME/attempt-049 \
  kneading/experiment/attempt-049/run_grid1000_seq12_prefixes_remap40_newmodel.sh
```

This generates a new set of PNGs/legends under the `_replot` tag and uploads
them to GCS. It does not recompute trajectories.

## Expected outputs

The run should produce:

- `grid1000_seq12_prefixes_remap40_newmodel_results.tsv`
- `grid1000_seq12_prefixes_remap40_newmodel_contours.png`
- `grid1000_seq12_prefixes_remap40_newmodel_prefix01_contours.png`
- `grid1000_seq12_prefixes_remap40_newmodel_prefix02_contours.png`
- `grid1000_seq12_prefixes_remap40_newmodel_prefix03_contours.png`
- `grid1000_seq12_prefixes_remap40_newmodel_prefix04_contours.png`
- `grid1000_seq12_prefixes_remap40_newmodel_prefix05_contours.png`
- `grid1000_seq12_prefixes_remap40_newmodel_prefix06_contours.png`
- `grid1000_seq12_prefixes_remap40_newmodel_prefix07_contours.png`
- `grid1000_seq12_prefixes_remap40_newmodel_prefix08_contours.png`
- `grid1000_seq12_prefixes_remap40_newmodel_prefix09_contours.png`
- `grid1000_seq12_prefixes_remap40_newmodel_prefix10_contours.png`
- `grid1000_seq12_prefixes_remap40_newmodel_prefix11_contours.png`
- `grid1000_seq12_prefixes_remap40_newmodel_prefix12_contours.png`
- matching `T` and `gamma` legend TSVs

## Attempt-049 plot defaults

The plotting defaults are intentionally different from `attempt-048`:

- `ATTEMPT049_CONTOUR_LINEWIDTH=0.35`
- `ATTEMPT049_PLOT_WIDTH=1600`
- `ATTEMPT049_PLOT_HEIGHT=1200`
- `ATTEMPT049_PLOT_PX_PER_UNIT=2.0`
- `ATTEMPT049_AXIS_LABEL_SIZE=34`
- `ATTEMPT049_AXIS_TITLE_SIZE=40`
- `ATTEMPT049_TICK_LABEL_SIZE=24`

These can be overridden for plot-only replay without recomputing trajectories.

## Final notes from the completed attempt-049 run

Attempt-049 completed the intended `1000 x 1000`, 12-symbol sweep on the GCE
VM and uploaded the final artifacts to:

```text
gs://carter-kneading-attempt048/attempt-049
```

Observed run facts:

- The VM used 24 visible Julia threads because quota forced the smaller
  machine size.
- The VM was configured with a `7h` max run duration and termination action
  `STOP`.
- The solve phase completed all `1000` checkpointed columns in roughly an
  hour.
- Final successful point count was `892665 / 1000000`.
- Final local retrieval produced `1041` files: `1000` checkpoint TSV columns,
  `13` PNGs, and the merged/log/legend TSV artifacts.
- The merged results TSV has `1000001` lines and is about `157M`.

Post-processing issue encountered:

- The first post-processing pass failed after all columns had completed because
  `contours.jl` incorrectly read raw-column `gamma_encoding` from `fields[12]`.
- Raw column files have `10` fields; `gamma_encoding` is `fields[7]`.
- Commit `4c2a3b4` fixed this indexing bug.
- Restarting the runner after that fix skipped all completed columns and only
  reran merge/plot/upload, confirming the checkpoint model works for this
  failure mode.

Cloud/result handling:

- After verifying final GCS artifacts, the VM was stopped to halt compute
  charges.
- A reusable boot image was created:

```text
attempt049-ready-20260429
```

- That image is separate from the result archive and can be used later for a
  new x86 VM, including a larger 96-vCPU VM, as long as GCS scopes/IAM are set
  correctly.
- The full retrieved `gcs_results` directory was packaged and uploaded to
  TGLFS with no password.
- TGLFS UFID:

```text
0fc24d9c3741c4febcfd1a623b66ad45e0a4c23eb51f24cb2e1ffa759a8be2f0
```

- Final TGLFS file-card name:

```text
MultimodalMaps.jl_kneading_experiment_attempt-049_grid1000_seq12_prefixes_remap40_newmodel_gcs_results.tar.gz
```

Practical lessons:

- Keep GCS as the authoritative handoff point before stopping the VM.
- Expect append-only logs to contain old failure text; use the newest
  timestamped run block and process state to interpret them.
- For future attempts, test post-processing on a tiny synthetic/raw-column TSV
  before launching a large scan, because the solve can succeed while merge or
  plotting still fails.
- Prefer clearly contextual artifact names that include repo, experiment area,
  attempt number, grid/sequence setup, and whether the archive is raw GCS
  results.
