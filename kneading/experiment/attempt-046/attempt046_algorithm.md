# attempt-046: Exact Contour/Coloring Rules

This document is the exact reproduction spec for the current `attempt-046`
interactive explorer.

It has two layers:

1. how the saved `attempt-027` data were generated from the Shimizu-Morioka
   flow and tangent dynamics
2. how `attempt-046` converts those saved data into the monotone-sign contour
   explorer and assigns contour colors

`attempt-046` does not reintegrate trajectories. It consumes the saved
`attempt-027` column files. So reproducing `attempt-046` exactly means:

- first reproduce the `attempt-027` saved event data
- then reproduce the `attempt-046` monotone-sign postprocessing and contour
  classification

## 1. Parameter Sweep and Stored Event Count

The saved source sweep is the `attempt-027` Shimizu-Morioka run with:

- `alpha` on a uniform grid of length `2000` over `[0.0, 0.7]`
- `lambda` on a uniform grid of length `2000` over `[0.2, 1.6]`
- `16` stored `|x|`-maximum events per parameter point
- nominal plot iterates `2:8`

The relevant source files are:

- [main.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-027/main.jl)
- [contours.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-027/contours.jl)
- [build_explorer.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-046/build_explorer.jl)

## 2. Shimizu-Morioka Flow Used in attempt-027

The ODE is

```text
x' = y
y' = x - lambda*y - x*z
z' = -alpha*(z - x^2)
```

with `B = 0`.

The tangent equation is the forward variational equation for a tangent vector
`v = (vx, vy, vz)`:

```text
vx' = vy
vy' = (1 - z)*vx - lambda*vy - x*vz
vz' = 2*alpha*x*vx - alpha*vz
```

## 3. Initial Condition and Initial Tangent

At each `(alpha, lambda)`:

### 3.1 Orbit initial condition

The orbit is seeded on one branch of the unstable manifold of the origin.

Define

```text
mu = 0.5 * (-lambda + sqrt(lambda^2 + 4))
e_u = (1, mu, 0)
u0 = eps0 * e_u / ||e_u||
eps0 = 1e-7
```

So the orbit starts at:

```text
u0 = eps0 * normalize((1, mu, 0))
```

This is exactly `unstable_side_initial_condition_025(lambda)`.

### 3.2 Tangent initial condition

The raw tangent seed is always:

```text
v_raw = (0, 0, 1)
```

This is not used directly. It is first orthogonalized against the flow and
then normalized.

Given the flow vector `f(u0)`, the orthogonalized tangent is:

```text
v = v_raw - ((v_raw . f) / (f . f)) * f
v = v / ||v||
```

provided `||v|| > 1e-12`.

This is exactly what `orthonormalize_against_flow_025` does.

## 4. ODE Solver Configuration in attempt-027

The combined orbit+tangent system is integrated with:

- solver: `Tsit5()`
- time span: `[0, 200]`
- adaptive stepping enabled
- `dt = 0.02`
- `dtmax = 0.02`
- `abstol = 1e-9`
- `reltol = 1e-9`
- `maxiters = 40000000`
- no saved dense output history

At every accepted solver step, the tangent returned by the solver is
re-orthogonalized against the current flow and renormalized before continuing.

## 5. How `|x|`-Maximum Events Are Detected

The event proxy is:

```text
g(u) = x*y
```

At a local extremum of `|x|`, one has `y = 0`, so `x*y = 0`.

The code declares a detected `|x|`-maximum event when:

```text
g_prev >= 0 and g_curr < 0
```

That is the precise sign-change test used by `scan_orbit_025`.

### 5.1 Time refinement

The raw crossing fraction is

```text
theta = clamp(-g_prev / (g_curr - g_prev), 0, 1)
```

If three successive states are available, the code refines the event time by a
quadratic vertex estimate on `x^2`.

Let

```text
z0 = x(t0)^2
z1 = x(t1)^2
z2 = x(t2)^2
```

using the previous-previous, previous, and current accepted solver times.
Then a parabola is fit through `(t0, z0)`, `(t1, z1)`, `(t2, z2)` and its
vertex time `t_vertex` is used if it lies in `[t0, t2]`.

The code then resets

```text
theta = clamp((t_vertex - t_prev) / (t_curr - t_prev), 0, 1)
```

### 5.2 Event state and event tangent

The event state is linearly interpolated:

```text
u_hit = (1 - theta) * u_prev + theta * u_curr
```

The event tangent is also linearly interpolated and then re-orthogonalized
against the flow at `u_hit`:

```text
v_hit_raw = (1 - theta) * v_prev + theta * v_curr
v_hit = orthonormalize_against_flow(v_hit_raw, u_hit, p)
```

## 6. What attempt-027 Stores at Each Event

For each detected event `k`, `attempt-027` stores:

- the full event state `(x_k, y_k, z_k)`
- the cumulative event time `T_k`
- the raw contour scalar

The raw contour scalar is

```text
d_k = sign(x_k) * (v_hit)_x
```

where `sign(x_k)` is taken as `+1` when `x_k >= 0` and `-1` otherwise.

This is exactly `signed_absx_component_025(u_hit, v_hit)`.

The TSV therefore stores:

- `absxmax_dot_values = [d_1, d_2, ..., d_16]`
- `absxmax_return_times = [T_1, T_2, ..., T_16]`
- `absxmax_states = [u_1, u_2, ..., u_16]`

Only results with status `"ok"` or `"short"` are used later.

## 7. How attempt-046 Rebuilds the Per-Iterate Grids

`attempt-046` reads the `attempt-027` column TSVs and builds:

- one dot grid per stored iterate
- one cumulative-time grid per stored iterate
- one interval-time grid per stored iterate

### 7.1 Interval return times

The cumulative times are converted to interval return times by:

```text
DeltaT_1 = T_1
DeltaT_k = T_k - T_{k-1}   for k >= 2
```

The explorer tables display interval times for iterates `2:8`.

## 8. Monotone-Sign Dots Used by attempt-046

The explorer does not contour the raw `d_k`. It constructs a new signed
sequence whose sign records whether the raw sign stayed the same or flipped.

Given the raw dot sequence `d_k`, define `m_k` as follows.

### 8.1 First stored event

If `d_1` is finite and nonzero:

```text
m_1 = |d_1|
```

If `d_1` is nonfinite or zero, that invalid value is kept.

### 8.2 Later stored events

For `k >= 2`, let `s_k = sign(d_k)` and let `s_prev` be the previous nonzero
raw sign.

Then:

```text
if s_k == s_prev:
    m_k = |d_k|
else:
    m_k = -|d_k|
```

and then `s_prev` is updated to `s_k`.

So the monotone sign at iterate `k` means:

- `+` if the raw dot-product sign stayed the same from the previous event
- `-` if the raw dot-product sign flipped from the previous event

This is exactly `monotone_sign_adjusted_dots_046`.

## 9. Which Iterates attempt-046 Uses

There are three different iterate ranges in play:

- stored source data: `1:16`
- plotted nominal iterates: `2:8`
- classification suffix limit: `2:12`

More precisely:

- the canvas only draws contours for nominal iterates `2, 3, ..., 8`
- contour geometry at nominal iterate `k` uses the monotone dots `m_k`
- color classification for a contour at nominal iterate `k` looks at a suffix
  starting at `k` and ending at `12`, but truncated by data availability

## 10. Square Evaluation Used by attempt-046

`attempt-046` deliberately disables the old skip-compression logic. Every
square is evaluated with zero local skip:

```text
skip = (0, 0, 0, 0)
```

So at nominal iterate `k`, every corner uses stored iterate `k` directly.

Corner order is always:

1. top-left
2. top-right
3. bottom-right
4. bottom-left

For a square `(j, i)` and nominal iterate `k`, the evaluator loads:

- current monotone dots at the four corners
- current interval times at the four corners
- next interval times at the four corners, meaning iterate `k + 1`

A square is:

- `missing` if any required iterate is unavailable, nonfinite, or zero-signed
- `constant` if all four current monotone signs match
- `mixed` otherwise

Only `mixed` squares are contoured.

## 11. Representative Corners for Comparing the Two Sides

For a mixed square, the code chooses exactly one representative negative corner
and one representative positive corner.

The selection is:

- scan corners in order `TL, TR, BR, BL`
- take the first negative corner as `neg_rep`
- take the first positive corner as `pos_rep`

Then compare the current interval times:

- if `current_time[neg_rep] < current_time[pos_rep]`, the negative side is the
  shorter-return-time side
- otherwise the positive side is the shorter-return-time side

The function returns:

- `shorter_sign`
- `short_rep`
- `long_rep`

This exact rule is `choose_representatives_025`.

## 12. Marching-Squares Geometry

All contour segments are zero contours of the current monotone dot field on the
given square.

### 12.1 Edge numbering

- edge `1`: top edge, `TL -> TR`
- edge `2`: right edge, `TR -> BR`
- edge `3`: bottom edge, `BR -> BL`
- edge `4`: left edge, `BL -> TL`

### 12.2 Edge intersection

On each edge, the endpoint offsets are:

```text
d1 = z1 - level
d2 = z2 - level
level = 0
```

If `d1` and `d2` have opposite signs, the intersection point is found by
linear interpolation:

```text
theta = d1 / (d1 - d2)
p = (1 - theta) * p1 + theta * p2
```

If one endpoint is exactly at level, the endpoint itself is used.

This is the exact interpolation used by `edge_point_025`.

### 12.3 Case index

The case index is:

```text
case_idx =
    (TL >= 0 ? 8 : 0) +
    (TR >= 0 ? 4 : 0) +
    (BR >= 0 ? 2 : 0) +
    (BL >= 0 ? 1 : 0)
```

### 12.4 Ambiguous cases

Cases `5` and `10` are resolved by the center average:

```text
center_value = 0.25 * (TL + TR + BR + BL)
```

Then:

- case `5`:
  - if `center_value >= 0`, pair `(1,2)` and `(3,4)`
  - else pair `(1,4)` and `(2,3)`
- case `10`:
  - if `center_value >= 0`, pair `(1,4)` and `(2,3)`
  - else pair `(1,2)` and `(3,4)`

### 12.5 Non-ambiguous case pairing table

The exact pairings are:

```text
1  -> (4,3)
2  -> (3,2)
3  -> (4,2)
4  -> (1,2)
6  -> (1,3)
7  -> (1,4)
8  -> (1,4)
9  -> (1,3)
11 -> (1,2)
12 -> (4,2)
13 -> (3,2)
14 -> (4,3)
```

Cases `0` and `15` emit nothing.

## 13. Sign Sequences Used for Color Classification

Color classification is not done from the four-corner square sign pattern
alone. It is done from two representative monotone-sign suffixes:

- one from the shorter-time representative corner
- one from the longer-time representative corner

For a contour at nominal iterate `k`, the code asks for signs on both corners
from iterates `k:k_end`, where:

```text
k_end = min(12, number_of_available_dot_grids)
```

### 13.1 Shared-suffix truncation

The current rule is:

- build the longest actually available suffix on each side
- truncate both to their common available length

This is important:

- a lack of later data no longer suppresses a mixed contour
- it only weakens the classification because fewer suffix entries are available

So a mixed square that has geometric contour segments is still drawn even if
the suffix is too short to prove a stronger color category.

## 14. Color Assignment Order

For each mixed square and nominal iterate `k`, color is assigned in this exact
order:

1. blue symbolic grazing, or blue return-time grazing when that mode is active
2. coordinate singularity test
3. purple symbolic grazing
4. real-contour test
5. fallback green

So:

- blue overrides everything
- red overrides purple/black/green
- purple overrides black/green
- black overrides green

## 15. Grazing, Symbolic Mode

This is the default mode in the explorer.

Let:

- `seq_a` = monotone-sign suffix on one representative side
- `seq_b` = monotone-sign suffix on the other representative side

The code requires `length(seq_a) == length(seq_b)`.

The deletion index range is:

```text
delete_idx = 1 : min(max(0, 9 - k + 1), length(seq_a) - 1)
```

So deletion is only allowed on the contour-relative iterate range `k:9`.

### 15.1 Blue symbolic grazing: delete one `+`

Blue is assigned if, for some `delete_idx`, either side can delete a `+` and
then match the other side exactly after shifting the remaining suffix left by
one place. Only the suffix from that deletion point onward matters.

Exact condition for candidate side `c` and other side `o`:

- `c[delete_idx] == +`
- entries before `delete_idx` match exactly
- for every later position:

```text
c[idx + 1] == o[idx]
```

### 15.2 Purple symbolic grazing: delete one `-` and flip the remaining suffix

Purple is assigned if, for some `delete_idx`, either side can delete a `-` and
then match the other side after inverting all later signs on the deleted side.
Only the suffix from that deletion point onward matters.

Exact condition:

- `c[delete_idx] == -`
- entries before `delete_idx` match exactly
- for every later position:

```text
-c[idx + 1] == o[idx]
```

The exact precedence inside the symbolic grazing scan is:

- scan `delete_idx` in ascending order
- at each `delete_idx`, test blue first
- only if blue fails at that same `delete_idx`, test purple

So the earliest matching deletion index wins, with blue taking precedence over
purple only at the same deletion index. The later red test can still override
purple.

## 16. Grazing, Return-Time Mode

The explorer also has a second grazing mode that reuses the old skip test from
`attempt-025` / `attempt-027`.

For the current mixed square evaluation:

- choose the shorter-time representative side and the longer-time
  representative side exactly as in Section 11
- define

```text
err_skip   = |DeltaT_short + DeltaT_short_next - DeltaT_long|
err_noskip = |DeltaT_short - DeltaT_long|
```

If

```text
err_skip < err_noskip
```

then the square is colored blue.

This is exactly `skip_increment_decision_025(evaluation)`.

### 16.1 Important caveat

This return-time grazing detector is not a first-principles monotone-sign
grazing condition. It is only the legacy one-miss skip heuristic reused as an
alternate blue classifier.

So the honest statement is:

- it makes sense if the goal is "show me where the old skip heuristic would
  have fired"
- it is not the same kind of object as the symbolic deletion rules
- it should be treated as a legacy comparison mode, not as a canonical grazing
  notion

## 17. Red: Coordinate Singularity

Red is checked after blue and before purple/black.

Only the local window `k:k+2` matters, meaning the first `3` entries of the
common suffix.

The exact condition is:

1. the two sequences differ in exactly two positions
2. those two positions are consecutive
3. on side `a`, the two differing entries are equal to each other
4. on side `b`, the two differing entries are equal to each other
5. the differing sign on side `a` is the opposite of the differing sign on
   side `b`

Equivalently, if the differing positions are `idx` and `idx+1`, then:

```text
seq_a[idx]     == seq_a[idx + 1]
seq_b[idx]     == seq_b[idx + 1]
seq_a[idx]     == -seq_b[idx]
seq_a[idx + 1] == -seq_b[idx + 1]
```

## 18. Black: Real Contour

Black is checked after blue, red, and purple.

Only the local window `k:k+1` matters, meaning the first `2` entries of the
common suffix.

Black is assigned exactly when those first two entries differ in exactly one
position.

## 19. Green: Fallback

Green means:

- the square is mixed at the nominal iterate
- marching squares found one or more contour segments
- the contour was not classified as grazing, coordinate singularity, or black

So green is not "no contour". Green is "contour exists, but it did not meet a
stronger classification rule."

## 20. Same-Edge Red-to-Black Suppression

`attempt-046` has one local follow-up suppression rule.

For each square, the code stores:

- the edge pairs emitted by the most recent red contour in that square
- the nominal iterate on which that red contour occurred

An edge pair is the unordered pair of intersected edges, packed as:

```text
pair_code = (min_edge << 4) | max_edge
```

### 20.1 What gets suppressed

Only this situation is suppressed:

- a red contour happened at iterate `k`
- a black contour happens at iterate `k+1`
- one of the black segments uses the exact same unordered edge pair as one of
  the red segments from iterate `k`

Those black segments are removed.

### 20.2 What does not get suppressed

The code does not suppress:

- black segments on different edge pairs
- blue segments
- purple segments
- green segments
- anything at iterates later than `k+1`

So this is a very narrow rule: "same square, next iterate only, black only,
same edge pair only."

## 21. Pseudocode for the attempt-046 Contour Pass

```text
build monotone dot grids from saved attempt-027 raw dot values
build cumulative time grids from saved attempt-027 event times
convert cumulative times to interval times

for each square (j, i):
    pending_red_pairs = empty
    pending_red_nominal = 0

    for nominal iterate k in 2:8:
        if pending_red_nominal != 0 and k > pending_red_nominal + 1:
            clear pending_red_pairs
            pending_red_nominal = 0

        evaluation = evaluate current square at iterate k with zero skip
        if evaluation is missing or constant:
            continue

        choose short and long representative corners by current interval time
        build common available monotone-sign suffixes on those two corners
        build zero-contour marching-squares segments from current monotone dots
        if no segments:
            continue

        classification =
            blue if the blue grazing test succeeds
            else red if coordinate singularity test succeeds
            else purple if the purple grazing test succeeds
            else black if real-contour test succeeds
            else green

        if classification is black and previous iterate had red in this square:
            remove black segments whose edge pair matches a pending red edge pair
            if no black segments remain:
                continue

        emit all surviving segments in their color bucket

        if classification is red:
            pending_red_pairs = the emitted red edge pairs
            pending_red_nominal = k
        else if pending_red_nominal + 1 == k:
            clear pending_red_pairs
            pending_red_nominal = 0
```

## 22. What the Tables in the Explorer Show

For sampled grid points, the explorer shows:

- raw sign sequence for iterates `2:8`
- monotone sign sequence for iterates `2:8`
- interval return times for iterates `2:8`
- the old skip column, which is intentionally always `no` in `attempt-046`

The displayed times are packed into the HTML after scalar quantization:

- the code chooses a scale from `(1000, 200, 100, 50, 10, 1)`
- each stored table value is `round(DeltaT_k * scale)`
- the displayed value is the dequantized value

So the tables reproduce the interval times approximately, not as raw
full-precision `Float64` values.

## 23. Reproduction Command

The current build/upload entrypoint is:

- [run_build_explorer_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-046/run_build_explorer_upload.sh)

That script:

1. reads the saved `attempt-027` sweep columns
2. rebuilds the monotone-sign explorer data
3. writes the self-contained HTML artifact
4. uploads it to TGLFS

## 24. Practical Interpretation

The shortest accurate summary is:

- `attempt-027` gives the raw `sign(x) * v_x` event data and event times
- `attempt-046` replaces the raw sign with a monotone flip/stay sign
- contours are geometric zero contours of the current monotone dot field
- colors are assigned from short-vs-long representative monotone-sign suffixes
- symbolic grazing is the primary sequence-based blue/purple rule
- return-time grazing is only the old skip heuristic reused as an alternate
  blue mode
