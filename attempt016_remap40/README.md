# attempt-016 remap40 package

This archive contains the Julia code needed to reproduce:

`kneading/experiment/attempt-016/grid500_seq7_prefixes_remap40_contours.png`

Run from the extracted archive root:

```bash
./run.sh
```

The package includes the repository-local source files used by the scan plus `Project.toml`
and `Manifest.toml` for the Julia package environment. A working Julia installation is
required. On first run, Julia may instantiate registered packages into the active depot.
The manifest was generated with Julia 1.12, so `run.sh` prefers `julia +release` when
Juliaup is available. The wrapper disables automatic full-environment precompile because
some optional visualization extensions are not needed by this scan. Override the Julia
command with `JULIA_CMD` if needed, for example:

```bash
JULIA_CMD="julia +1.12" ./run.sh
```

The default run is the original `500 x 500`, `MAX_SEQ_LENGTH=7`, remap-resolution-40
scan. It writes outputs under `kneading/experiment/attempt-016/`.
