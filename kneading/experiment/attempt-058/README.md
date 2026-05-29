# attempt-058

Single local trajectory at `g_h = 0`, `Delta Ca = -35`, `Delta x = -1` with the `y` variable stubbed out identically to zero.

The script compares convergence of:

- the leading Lyapunov exponent, converted from natural-log units per model time unit to bits/s using `1000 / log(2)`;
- Abramov-normalized LZ76 complexity of the detected SSCS, computed as `phrase_count * log2(n_symbols) / elapsed_seconds`, where `elapsed_seconds` is the time of the last used SSCS event after the discarded transient.

Default run:

```sh
julia --project=. kneading/experiment/attempt-058/main.jl
```

Smoke test override:

```sh
ATTEMPT058_RUN_T=1e4 ATTEMPT058_TRANSIENT_T=1e3 julia --project=. kneading/experiment/attempt-058/main.jl
```
