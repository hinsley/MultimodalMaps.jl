# attempt-026: Permanent Square Retirement After First Skip

## Summary

`attempt-026` is a plotting-only update to `attempt-025`.

It uses the same saved `|x|`-max dataset from `attempt-025`, the same
flow-orthogonal tangent dot products, and the same delta-time skip logic:

- stored cumulative hit times `T_k`
- plotting-time interval conversion
  - `Δt_1 = T_1`
  - `Δt_k = T_k - T_{k-1}` for `k >= 2`
- the same one-miss test
  - `err_skip = |Δt_cur_short + Δt_next_short - Δt_cur_long|`
  - `err_noskip = |Δt_cur_short - Δt_cur_long|`

The algorithm change is purely in what happens **after** a skip increment is
detected in a square.

## Difference From attempt-025

In `attempt-025`, a mixed-sign square that satisfied the skip test behaved like
this:

1. Draw the pre-increment contour as an excluded contour when debug output was
   enabled.
2. Increment the local skip counters on the shorter-time sign side.
3. Re-evaluate the square immediately.
4. If the re-evaluated square was still mixed-sign, emit an accepted contour in
   that same nominal iterate.
5. On later nominal iterates, the same square could continue to emit accepted
   contours using the updated skip counters.

That means a square could first appear in red as “pruned” and later still
contribute white accepted contours.

`attempt-026` removes that behavior.

## New Rule

Once a square triggers a skip increment, that square is **retired permanently**
for the rest of the plot.

Concretely:

1. Evaluate the square with its current local skip counters.
2. If the square is mixed-sign and the skip test passes:
   - draw the current pre-increment contour in red if that nominal iterate is
     inside the requested overlay range
   - increment the local skip counters on the shorter-time sign side
   - mark the square as retired
   - do **not** draw a white accepted contour in that square for the current
     iterate
   - do **not** re-evaluate the square for contour emission
3. On all later nominal iterates, if the square is already retired:
   - skip it immediately
   - draw neither white nor red contours in it

So the operational interpretation is:

- a red contour is the final visible output for that square
- after the first skip event, the square is done forever

## Data And State

`attempt-026` keeps all `attempt-025` data assumptions:

- saved `|x|`-max dot values
- saved cumulative hit times
- plotting-time conversion to interval times
- local duplicated per-square corner skip counters
- deterministic representative-corner selection
- ordinary linear marching squares on the currently evaluated field

It adds one new per-square state field:

- `retired_square[j, i] :: Bool`

This is independent of the four local corner skip counters. The skip counters
still update when the one-miss test fires, but the square-retirement mask is
what prevents any future white contour emission in that square.

## Pseudocode

```text
procedure PROCESS_SQUARE_AT_ITERATE_ATTEMPT026(Q, n, dataset, skip_state, retired_square):
    if retired_square[Q]:
        return no accepted segments, no excluded segments

    evaluation = EVALUATE_SQUARE(Q, n, dataset, skip_state)

    if evaluation.status == missing_data:
        return no accepted segments, no excluded segments

    if evaluation.status == constant_sign:
        return no accepted segments, no excluded segments

    shorter_sign = DETERMINE_SHORTER_TIME_SIGN(evaluation)
    err_skip, err_noskip = COMPUTE_SKIP_ERRORS(evaluation, shorter_sign)

    if err_skip < err_noskip:
        excluded_segments = MARCH_CURRENT_SQUARE(evaluation.current_dot)
        INCREMENT_SHORTER_SIGN_SKIP_COUNTERS(Q, evaluation.sign, shorter_sign, skip_state)
        retired_square[Q] = true
        return no accepted segments, excluded_segments

    accepted_segments = MARCH_CURRENT_SQUARE(evaluation.current_dot)
    return accepted_segments, no excluded segments
```

## Consequences

Compared with `attempt-025`, `attempt-026` has these consequences:

- fewer white accepted contours overall
- the first detected skip event in a square becomes terminal for that square
- red contours represent the last visible activity in retired squares
- no later accepted contour can appear in a square whose local skip process has
  already fired once

## Intended Output For The Requested Plot

The requested `attempt-026` PNG:

- reuses the completed `attempt-025` sweep data
- processes nominal iterates `1:8` to evolve the square-retirement state
- overlays only nominal iterates `2:8`
- uses a black background
- uses white for accepted contours
- uses red for excluded contours
- uses the thicker line width carried over from the latest `attempt-025` overlay
