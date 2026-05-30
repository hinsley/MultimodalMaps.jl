# Rossler y-minima tangent scan 4096 artifacts

This folder contains the 4096 x 4096 Rossler y-minima tangent scan output for:

- `b = 0.3`
- `c in [2.0, 7.0]`
- `a in [0.30, 0.55]`
- 8 y-minima events after transients
- max integration time `450.0`

The raw local TSV gzip is `coarse_scan.tsv.gz`. It preserves the full row schema:

```text
a c b status events word code period gamma max_time first_time last_time min_y max_y
```

The `word` field is the canonical 8-symbol sequence. The monotone-sign sequence is derived from adjacent symbol equality/flips and is not a replacement for the saved symbol sequence.

## GitHub-size packaging

Some raw artifacts exceed GitHub's normal 100 MB per-file limit, so they are committed in reconstructable forms:

- `coarse_scan.tsv.gz` is committed as `coarse_scan.tsv.gz.part-00` through `coarse_scan.tsv.gz.part-05`.
- The raw standalone probe HTML files are committed as gzip files:
  - `contours/coarse_scan_8bit_word_heatmap_probe.html.gz`
  - `contours/coarse_scan_7bit_monotone_heatmap_probe.html.gz`

Reconstruct the full TSV gzip:

```bash
cat coarse_scan.tsv.gz.part-* > coarse_scan.tsv.gz
shasum -a 256 -c SHA256SUMS --ignore-missing
```

Restore the standalone HTML probes:

```bash
gzip -dk contours/coarse_scan_8bit_word_heatmap_probe.html.gz
gzip -dk contours/coarse_scan_7bit_monotone_heatmap_probe.html.gz
```

The PNG heatmaps and legend TSVs are committed directly under `contours/`.

## Verification

The raw local artifacts were verified with:

```bash
python3 flow_folding/examples/verify_rossler_y_minima_tangent_outputs.py
```

Verifier result:

- TSV rows: `16,777,216`
- completed 8-symbol words: `12,676,972`
- max-time-limited rows: `4,100,244`
- PNG dimensions: `25600 x 17600`
- word palette legend rows: `256`
- monotone palette legend rows: `128`
- probe code bytes: `16,777,216`
- probe valid bits: `2,097,152`
