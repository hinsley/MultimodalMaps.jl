# attempt-051 filtering notes

The final figure is generated from the saved TSV, not by recomputing orbits.
For each `g_h` case, the filtered replay reads:

```text
grid1000_seq12_tmax1e5_<gh-label>_prefixes_remap40_newmodel_results.tsv
```

and writes:

```text
grid1000_seq12_tmax1e5_<gh-label>_prefixcompatible_tzero2to12_contours.png
```

Filtering rules are inherited from the final `attempt-050` plot, with the
explicit `attempt-051` clarification that blue symbol 1 is retained:

- Red `T` categories are encoded from `T_scs[2:end]`, so red contours never
  come from changes in the first T symbol.
- Red marched squares are omitted when all four corners have `T_scs[2:12]`
  present and equal to zero.
- Blue `Gamma_SD^-` categories use the full saved `gamma_scs`, including
  symbol 1.
- Red and blue both suppress marched squares whose four corner sequences are
  prefix-compatible after truncating longer corner sequences to the shortest
  nonempty corner sequence.

Opacity follows the current attempt-050 convention: the first visible mismatch
in a plotted family has alpha `1`; later mismatch layers use `1 / k^0.3`.
Because red drops T symbol 1 before building its sequence matrix, red mismatch
layer 1 corresponds to original T symbol 2.
