# attempt-057

`attempt-057` scans one deterministic arbitrary initial condition over the
same parameter window as `attempt-050`, rather than using the T/gamma critical
initial conditions.

Default scan settings:

- grid `500 x 500`
- `Delta Ca in [-45, -20]`
- `Delta x in [-1.5, -0.5]`
- `max_seq_length = 20`
- SSCS hard integration limit `tmax = 3e5`
- initial condition rule:
  `V = -30`, `x = xinf(p,V) - 1e-4`, `y = yinf(V)`, `n = ninf(V)`,
  `h = hinf(V)`, `Ca = Ca_null_Ca(p,V)`

Each row saves:

- `arbitrary_scs`
- absolute `event_times` for detected SSCS symbols
- `event_intervals`, where interval `i` is the elapsed integration time since
  the previous detected SSCS symbol, with the first interval measured from
  `t = 0`

The GCS runner processes `g_h` values sequentially in this order:

1. `0.0`
2. `0.001`
3. `0.01`

After each `g_h` finishes, its columns, merged TSV, summary, and log are
uploaded to `gs://carter-kneading-attempt057/attempt-057/` before the next
`g_h` starts.
