# attempt-025: Skip-Adjusted `|x|`-Max Contour Algorithm

## Summary

`attempt-025` should **recompute the dataset from scratch** and then plot from
that newly computed dataset. It should not rely on the existing `attempt-024`
column files as the final data source, because those files do not store the
`|x|`-max return times required by the new contouring rule.

The recomputation stage should reuse the exact event-generation mechanics that
were correct in `attempt-024`:

- same Shimizu-Morioka flow
- same tangent equation
- same unstable-manifold initial seed
- same flow-orthogonal tangent renormalization
- same `|x|`-max detection logic
- same interpolation/refinement logic at event hits

The only required data-model extension is that `attempt-025` must also record
the `|x|`-max return times explicitly, in addition to the tangent dot products
and event states.

After recomputation, the contour plot should be built from a new
square-local, per-corner skipped-iterate algorithm that suppresses fake
contours caused by locally missing one `|x|`-maximum as parameters vary.

## Goals

1. Recompute a correct `|x|`-max dataset from scratch, with return times stored.
2. Support up to **16** `|x|`-max iterates in the computed dataset.
3. Allow plotting as if only the first `M` iterates existed, for any
   `1 <= M <= 16`, by changing one plotting cap parameter.
4. Use the square-local skipped-iterate rule to correct fake contours caused by
   one missed `|x|`-maximum.
5. Keep the contour geometry based on ordinary linear marching-squares
   interpolation after the local skip adjustment is applied.

## Why recomputation is required

The current `attempt-024` `|x|`-max outputs contain:

- `absxmax_dot_values`
- `absxmax_states`

They do **not** contain:

- `absxmax_return_times`

The new plotting algorithm requires, for each effective iterate used at a
corner:

- current dot value
- current return time
- next return time

Therefore the `attempt-025` pipeline must recompute the `|x|`-max event
sequence from scratch and store the return times explicitly.

## Exact recomputation stage

This stage should match `attempt-024` exactly wherever possible, except for:

- increasing the stored `|x|`-max iterate support to 16
- writing `|x|`-max return times to disk
- optionally omitting any unrelated `z_max` outputs if they are not needed

### Parameter grid and solver defaults

Reuse the same default parameter-domain and integration settings from
`attempt-024`, unless there is a deliberate later reason to change them:

- `N_alpha = 1200`
- `N_lambda = 1200`
- `alpha ∈ [0.0, 0.7]`
- `lambda ∈ [0.2, 1.6]`
- `max_event_iterates = 16`
- `t_end = 200.0`
- `dt = 0.02`
- `abstol = 1e-9`
- `reltol = 1e-9`
- `maxiters = 40_000_000`
- `eps0 = 1e-7`
- `max_state = 200.0`
- `tangent_eps = 1e-12`

### Flow and tangent equations

Use the same Shimizu-Morioka flow as in `attempt-024`.

Flow state `u = (x, y, z)` with parameters `(alpha, lambda)`:

\[
\dot{x} = y
\]
\[
\dot{y} = x - \lambda y - x z
\]
\[
\dot{z} = -\alpha (z - x^2)
\]

Use the same tangent equation on `v = (v_x, v_y, v_z)` as in `attempt-024`:

\[
\dot{v}_x = v_y
\]
\[
\dot{v}_y = (1-z)v_x - \lambda v_y - x v_z
\]
\[
\dot{v}_z = 2 \alpha x v_x - \alpha v_z
\]

### Initial conditions

For each parameter point `(alpha, lambda)`:

1. Compute the unstable-side seed using the exact same formula as
   `attempt-024`:

   \[
   \mu = \frac{-\lambda + \sqrt{\lambda^2 + 4}}{2}
   \]

   \[
   \text{vec} = (1, \mu, 0)
   \]

   \[
   u_0 = \varepsilon_0 \cdot \frac{\text{vec}}{\|\text{vec}\|}
   \]

2. Initialize the tangent direction with the same vector used in `attempt-024`:

   \[
   v_{\text{init}} = (0, 0, 1)
   \]

3. Orthogonalize that tangent against the flow at `u_0`, then normalize it.
   This is mandatory and should use the exact same logic as `attempt-024`.

### Flow-orthogonal tangent normalization

At every accepted solver step, and again at every interpolated event hit:

1. Compute the flow vector `f(u)`.
2. Remove the component of the tangent parallel to the flow:

   \[
   v \leftarrow v - \frac{v \cdot f}{f \cdot f} f
   \]

   only when `f ⋅ f > tangent_eps^2`.

3. Normalize the resulting vector.
4. If normalization fails or any quantity becomes nonfinite, terminate the
   orbit with status `"bad_tangent"` or `"nonfinite"` as appropriate.

This is exactly the `attempt-024` behavior and should not be altered.

### `|x|`-max event definition

Use the exact same proxy and hit logic that `attempt-024` used for `|x|`-max
events.

Define the proxy:

\[
\text{absxmax\_proxy}(u) = x y
\]

Track an event when:

- previous proxy value is `>= 0`
- current proxy value is `< 0`

This is the same crossing rule as `attempt-024`.

### Event interpolation and quadratic refinement

When an `|x|`-max event is detected:

1. Compute the default crossing fraction:

   \[
   \theta = \text{clamp}\left(\frac{-p_{\text{prev}}}{p_{\text{curr}} - p_{\text{prev}}}, 0, 1\right)
   \]

   with the exact `attempt-024` fallback `0.5` if the denominator is zero.

2. If a previous-previous sample exists, refine the hit time using the same
   quadratic vertex logic used in `attempt-024`, but applied to `x^2`:

   - use the three times `(t_prevprev, t_prev, t_curr)`
   - use the three values `(x_prevprev^2, x_prev^2, x_curr^2)`
   - compute the quadratic vertex time
   - if valid, convert it back to a local interpolation fraction

3. Interpolate the state linearly between `u_prev` and `u_curr`.
4. Interpolate the tangent linearly between `v_prev` and `v_curr`.
5. Orthogonalize the interpolated tangent against the flow at the interpolated
   hit state, then normalize it.

### Values to record at each `|x|`-max event

For each detected `|x|`-max event, store:

1. **Return time**
   - `t_hit`
2. **Contour scalar**
   - signed `x`-component of the flow-orthogonal tangent:

     \[
     \text{dot} =
     \begin{cases}
     v_x & \text{if } x_{\text{hit}} \ge 0 \\
     -v_x & \text{if } x_{\text{hit}} < 0
     \end{cases}
     \]

3. **Full event state**
   - `(x_hit, y_hit, z_hit)`

This preserves the exact tangent-derived quantity from `attempt-024` while
adding the missing return times.

### Stop condition and statuses

Stop integrating a parameter point when either:

- `t >= t_end`, or
- 16 `|x|`-max events have been recorded, or
- failure occurs

Use the same status-style as `attempt-024`:

- `"ok"`: all requested events were recorded
- `"short"`: integration ended before 16 events were found
- `"blowup"`: state norm exceeded `max_state`
- `"bad_tangent"`: tangent orthogonalization/normalization failed
- `"nonfinite"`: nonfinite state or tangent encountered

## Output data model

Use a column-sharded TSV layout like earlier attempts.

### Per-column TSV schema

Recommended header:

```text
alpha	lambda	absxmax_count	absxmax_dot_values	absxmax_return_times	absxmax_states	status
```

Field meanings:

- `alpha`
- `lambda`
- `absxmax_count`
- `absxmax_dot_values`
  - comma-separated list of up to 16 floats
- `absxmax_return_times`
  - comma-separated list of up to 16 floats
- `absxmax_states`
  - semicolon-separated triples `x,y,z`
- `status`

### Merged TSV

Also write a merged TSV with the same schema after all columns are complete.

### Why states are still stored

Even though the new contouring rule only requires dots and return times, keep
storing full states because:

- they were already part of the correct `attempt-024` event generation
- they preserve future debugging flexibility
- they allow later extraction of explicit `x` and `z` iterate values without
  recomputation

## Plotting-stage parameters

Define two separate iterate bounds:

- `MAX_AVAILABLE_ITERATES = 16`
- `MAX_PLOTTED_ITERATES`, with `1 <= MAX_PLOTTED_ITERATES <= 16`

The recomputed dataset should support all 16 iterates.

The plotter should only process nominal iterates:

\[
1, 2, \dots, \text{MAX\_PLOTTED\_ITERATES}
\]

This is how the same recomputed dataset can be rendered as if only the first 8
iterates existed.

## Square-local skipped-iterate algorithm

## Core idea

The plotted contour at nominal iterate `n` is not based directly on the stored
iterate-`n` values at every corner. Instead, each corner-instance of each
marched square carries a local nonnegative integer skip counter.

At nominal iterate `n`, a corner with skip counter `s` reads from effective
iterate:

\[
k = n + s
\]

This lets one side of a square locally advance by one iterate when the timing
test indicates that a single `|x|`-maximum was missed.

## Per-square state

Each marched square owns 4 duplicated corner-local counters:

- `skip_tl`
- `skip_tr`
- `skip_br`
- `skip_bl`

Important:

- these counters are **not shared** with adjacent squares
- geometrically coincident corners in neighboring squares may carry different
  counter values
- uninitialized counters are treated as zero

## Generic assumptions

Assume all of the following:

1. No effective dot value is exactly zero at a corner.
2. In a mixed-sign square, the sign partition matches the shorter-time versus
   longer-time partition.
3. The grid is fine enough that at most one `|x|`-maximum is missed in a single
   square during a single nominal iterate.
4. Squares are processed independently; no connected-component logic is used.

## Square evaluation rule

For a fixed nominal iterate `n` and a fixed square:

1. Read the square's 4 skip counters.
2. Compute the 4 effective iterate indices:
   - `k_tl = n + skip_tl`
   - `k_tr = n + skip_tr`
   - `k_br = n + skip_br`
   - `k_bl = n + skip_bl`
3. At each corner, fetch:
   - current dot `dot[k]`
   - current return time `time[k]`
   - next return time `time[k+1]`
4. If any required lookup fails at any corner:
   - abandon this square for nominal iterate `n`
   - draw nothing
   - do not update counters
5. If the 4 effective dot values are constant-sign:
   - draw nothing
   - do not update counters
6. If the square is mixed-sign:
   - determine which sign side is the shorter-return-time side
   - run the one-miss test
7. If the one-miss test passes:
   - increment by `1` the skip counter of every corner in the square whose
     current sign belongs to the shorter-time side
   - then re-evaluate the square exactly once using the updated counters
8. Do not repeat the skip test again for that square during that nominal
   iterate.
9. If the final re-evaluated square is data-complete and mixed-sign:
   - contour it using standard linear interpolation on the final effective dot
     values

## Deterministic representative-corner convention

The user-level rule is that any positive/negative representative pair is fine
generically. For implementation determinism, fix the corner order:

```text
tl, tr, br, bl
```

Whenever a representative corner is needed:

- choose the first negative corner in that order
- choose the first positive corner in that order

This is only a determinism concession. It is not meant to change the intended
generic mathematics.

## One-miss test

Let:

- `c_short` be a representative corner on the shorter-return-time sign side
- `c_long` be a representative corner on the longer-return-time sign side

Define:

- `t_cur_short = time[c_short][k_cur_short]`
- `t_next_short = time[c_short][k_cur_short + 1]`
- `t_cur_long = time[c_long][k_cur_long]`

Compute:

\[
\text{err\_skip} = |t_{\text{cur,short}} + t_{\text{next,short}} - t_{\text{cur,long}}|
\]

\[
\text{err\_noskip} = |t_{\text{cur,short}} - t_{\text{cur,long}}|
\]

If:

\[
\text{err\_skip} < \text{err\_noskip}
\]

then interpret that as evidence that the shorter-time side is one iterate
behind because it has not missed the event that the longer-time side has
already skipped.

In that case:

- increment the skip counters of all corners on the shorter-time sign side
- then recompute the square once and contour from the **updated** effective dot
  values, not the old ones

## Marching-squares rule

After the optional one-time skip increment and re-evaluation:

- use standard marching squares on the final effective dot values
- use ordinary linear interpolation on each crossed edge
- reuse the same ambiguous-case convention as `attempt-024`

For an edge with endpoint values `d1` and `d2`, use:

\[
\theta = \frac{d_1}{d_1 - d_2}
\]

and linearly interpolate the edge point with that `theta`.

The algorithm changes **which** values get contoured, not **how** the contour
point on an edge is placed.

## Coloring convention

Color contours by **nominal iterate**, not by the effective shifted iterate
used at each corner.

This is a deliberate concession for simplicity. A single square may be drawn
from corner values coming from different effective iterates after local skip
updates, but the plotted color should still be tied to the outer-loop iterate
number.

## Full pseudocode

### Top-level pipeline

```text
procedure RUN_ATTEMPT025():
    dataset = RECOMPUTE_ABSXMAX_DATASET()
    write column TSVs
    write merged TSV

    plot_cap = clamp(user_plot_cap, 1, 16)
    square_state = INITIALIZE_ZERO_SKIP_STATE_FOR_ALL_SQUARES()
    emitted_segments = empty list keyed by nominal iterate

    for nominal_iterate in 1:plot_cap:
        PROCESS_NOMINAL_ITERATE(nominal_iterate, dataset, square_state, emitted_segments)

    render emitted_segments using nominal-iterate colors
    write debug artifacts if enabled
```

### Recompute dataset

```text
procedure RECOMPUTE_ABSXMAX_DATASET():
    for each alpha column:
        for each lambda row:
            result = SCAN_ORBIT(alpha, lambda)
            append result to column TSV
    merge columns into one TSV
```

### Orbit scan

```text
procedure SCAN_ORBIT(alpha, lambda):
    p = (alpha, lambda)

    mu = 0.5 * (-lambda + sqrt(lambda^2 + 4))
    unstable_vec = normalize((1, mu, 0))
    orbit_u0 = eps0 * unstable_vec

    tangent_seed = (0, 0, 1)
    tangent_u0 = ORTHOGONALIZE_AND_NORMALIZE(tangent_seed, orbit_u0, p)
    if tangent_u0 fails:
        return bad_tangent result

    initialize Tsit5 integrator exactly as in attempt-024

    dot_values = empty vector
    return_times = empty vector
    states = empty vector
    status = "ok"

    u_prev = orbit_u0
    v_prev = tangent_u0
    t_prev = 0
    proxy_prev = x_prev * y_prev
    have_prevprev = false

    while t < t_end and length(dot_values) < 16:
        step integrator

        u_curr = current state
        v_curr_raw = current tangent
        v_curr = ORTHOGONALIZE_AND_NORMALIZE(v_curr_raw, u_curr, p)
        if fail:
            status = "bad_tangent"
            break

        overwrite integrator tangent state with v_curr

        if nonfinite state or tangent:
            status = "nonfinite"
            break

        if max(abs(u_curr)) > max_state:
            status = "blowup"
            break

        proxy_curr = x_curr * y_curr

        if proxy_prev >= 0 and proxy_curr < 0:
            theta = crossing fraction from proxy sign change

            if have_prevprev:
                compute quadratic vertex time using x^2 at
                    (t_prevprev, t_prev, t_curr)
                if valid:
                    replace theta by the vertex-based fraction

            u_hit = linear interpolation of state at theta
            v_hit_raw = linear interpolation of tangent at theta
            v_hit = ORTHOGONALIZE_AND_NORMALIZE(v_hit_raw, u_hit, p)
            if fail:
                status = "bad_tangent"
                break

            t_hit = t_prev + theta * (t_curr - t_prev)
            dot_hit = sign(x_hit) * v_hit_x

            append dot_hit to dot_values
            append t_hit to return_times
            append u_hit to states

        shift prevprev <- prev
        shift prev <- curr

    if status == "ok" and length(dot_values) < 16:
        status = "short"

    return result(alpha, lambda, count, dot_values, return_times, states, status)
```

### Process one nominal iterate

```text
procedure PROCESS_NOMINAL_ITERATE(n, dataset, square_state, emitted_segments):
    for each marched square Q independently:
        segments = PROCESS_SQUARE_AT_ITERATE(Q, n, dataset, square_state[Q])
        append segments to emitted_segments[n]
```

### Process one square at one nominal iterate

```text
procedure PROCESS_SQUARE_AT_ITERATE(Q, n, dataset, skip_state):
    evaluation = EVALUATE_SQUARE(Q, n, dataset, skip_state)

    if evaluation.status == missing_data:
        return no segments

    if evaluation.status == constant_sign:
        return no segments

    shorter_sign = DETERMINE_SHORTER_TIME_SIGN(evaluation)
    (short_rep, long_rep) = CHOOSE_REPRESENTATIVES(evaluation, shorter_sign)

    err_skip =
        abs(
            evaluation.current_time[short_rep]
            + evaluation.next_time[short_rep]
            - evaluation.current_time[long_rep]
        )

    err_noskip =
        abs(
            evaluation.current_time[short_rep]
            - evaluation.current_time[long_rep]
        )

    if err_skip < err_noskip:
        for corner in {tl, tr, br, bl}:
            if evaluation.sign[corner] == shorter_sign:
                skip_state[corner] += 1

        evaluation = EVALUATE_SQUARE(Q, n, dataset, skip_state)

        if evaluation.status == missing_data:
            return no segments

        if evaluation.status == constant_sign:
            return no segments

    return EMIT_MARCHED_SEGMENTS(Q, evaluation.final_dot_values)
```

### Evaluate a square

```text
procedure EVALUATE_SQUARE(Q, n, dataset, skip_state):
    for corner in {tl, tr, br, bl}:
        k[corner] = n + skip_state[corner]

        require dot[corner][k[corner]]
        require time[corner][k[corner]]
        require time[corner][k[corner] + 1]

        if any required value is missing:
            return missing_data

        current_dot[corner] = dot[corner][k[corner]]
        current_time[corner] = time[corner][k[corner]]
        next_time[corner] = time[corner][k[corner] + 1]
        sign[corner] = sign(current_dot[corner])

    if all signs are identical:
        return constant_sign payload

    return mixed_sign payload
```

### Determine shorter-time sign side

```text
procedure DETERMINE_SHORTER_TIME_SIGN(evaluation):
    neg_rep = first corner in order (tl, tr, br, bl) with negative sign
    pos_rep = first corner in order (tl, tr, br, bl) with positive sign

    if evaluation.current_time[neg_rep] < evaluation.current_time[pos_rep]:
        return negative
    else:
        return positive
```

### Emit final contour segments

```text
procedure EMIT_MARCHED_SEGMENTS(Q, final_dot_values):
    run the same marching-squares case logic as attempt-024
    use linear interpolation on crossed edges
    return zero, one, or two line segments
```

## Optimization notes

This section must be treated as part of the implementation spec.

### Recompute-stage optimizations

1. **Column sharding with resume**
   - write one TSV per fixed-alpha column
   - skip columns already complete
2. **Stop once 16 `|x|`-max events are found**
   - do not keep integrating past the required event budget
3. **No saved step history**
   - keep `save_everystep=false`, `save_start=false`, `save_end=false`
4. **Only three time levels kept in memory**
   - current, previous, previous-previous
5. **In-place tangent overwrite**
   - after orthogonalizing `v_curr`, write it back into the integrator state
6. **Pre-sized event arrays**
   - preallocate vectors for 16 events

### Plot-stage optimizations

1. **Plot cap**
   - `MAX_PLOTTED_ITERATES` lets the same 16-iterate dataset render as if only
     the first `M` iterates existed
2. **Sparse skip-state storage**
   - only materialize square skip states once a square first needs nonzero
     counters
3. **Early missing-data exit**
   - abandon a square immediately if any required lookup is missing
4. **Early constant-sign exit**
   - do not run the skip test on constant-sign squares
5. **Single re-evaluation only**
   - after one counter increment, re-evaluate once and stop
6. **Representative-corner reduction**
   - once sign partition is known, the skip test uses only one representative
     per sign side
7. **No neighbor synchronization**
   - duplicated per-square corners avoid expensive cross-square updates
8. **Final-only segment emission**
   - never keep transient pre-increment contour segments
9. **Reuse of `attempt-024` marching-squares cases**
   - do not invent a new edge-topology rule

### Intentional non-optimizations

These are deliberate omissions and should stay omitted unless a later attempt
changes the design:

- no tangency-score masking
- no component-level merging or pruning
- no sharing of skip counters between neighboring squares
- no multiple skip corrections in the same square during one nominal iterate

## Debug artifacts to write

Recommended optional artifacts:

1. per-square final skip counters:
   - `skip_tl`, `skip_tr`, `skip_br`, `skip_bl`
2. per-square per-iterate decision log:
   - effective iterate indices
   - representative corners
   - current times
   - next time on shorter side
   - `err_skip`
   - `err_noskip`
   - increment flag
   - final square outcome
3. emitted contour segments grouped by nominal iterate
4. diagnostic plot of incremented squares only
5. summary counts:
   - missing-data squares
   - constant-sign squares
   - incremented squares
   - finally contoured squares

## Acceptance checks

The eventual implementation should satisfy all of these:

1. A constant-sign square produces no contour and no counter update.
2. A mixed-sign square with complete data and `err_skip >= err_noskip`
   contours from the current effective values without changing counters.
3. A mixed-sign square with complete data and `err_skip < err_noskip`
   increments only the shorter-side sign corners, re-evaluates once, and
   contours from the updated values if still mixed-sign.
4. A square that becomes constant-sign after the increment produces no contour.
5. A square with any missing required lookup produces no contour.
6. Counter values persist from nominal iterate `n` to `n+1` for the same
   square.
7. Geometrically coincident corners in neighboring squares may have different
   skip counters.
8. Rendering with plot cap `8` uses only nominal iterates `1:8` even when the
   dataset contains 16 iterates.
9. Contour color remains tied to nominal iterate, not effective iterate.

## Final implementation defaults

- recompute data from scratch
- use exact `attempt-024` event-generation mechanics for `|x|`-maxima
- extend stored data to include `absxmax_return_times`
- support 16 stored iterates
- allow plotting with any cap from 1 to 16
- use nominal-iterate coloring
- use duplicated per-square corner skip counters
- use one skip test and at most one increment per square per nominal iterate
- use standard linear marching squares on the final effective dot values
