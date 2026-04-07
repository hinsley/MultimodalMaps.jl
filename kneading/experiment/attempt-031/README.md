# Attempt 031

This attempt replaces the tangent-vector / orthonormalization contour scalar from [attempt-027](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-027/README.md) with a forward-sensitivity scalar computed by `SciMLSensitivity.jl`.

## Goal

For each parameter point `(alpha, lambda)` in the Shimizu-Morioka sweep:

- seed one unstable-manifold branch of the origin equilibrium exactly as before
- integrate once only
- harvest successive local `|x|`-maxima from that single trajectory
- save the signed forward sensitivity
  \[
  \frac{\partial}{\partial z_0}\left(x_{\mathrm{return}}^2\right)
  \]
  at each detected `|x|`-maximum return

Because the saved value is a real sensitivity rather than just a sign, the same dataset can support:

- zero-contour overlays of signed sensitivities
- single-iterate heatmaps of sensitivity magnitude `abs(value)`

## Detection model

This attempt uses the exact `y = 0` section, not the old `x*y` proxy.

- `x' = y`, so extrema of `x^2` occur when `y = 0`
- at `y = 0`,
  \[
  \frac{d^2}{dt^2}(x^2)=2x^2(1-z)
  \]
- therefore `z > 1` identifies local maxima of `x^2`, i.e. local `|x|`-maxima

So a return is kept iff:

- the trajectory crosses `y = 0`
- `abs(x)` at that crossing is safely away from `0`
- `z > 1` at that crossing

## Sensitivity model

The varied quantity is the initial `z` coordinate only. This attempt does **not** evolve a tangent vector manually.

Instead:

- the physical initial condition is
  \[
  X_0(\varepsilon)=X_{\mathrm{unstable}} + (0,0,\varepsilon)
  \]
- `SciMLSensitivity.ODEForwardSensitivityProblem` is used with one sensitivity parameter `\varepsilon`
- the forward sensitivity returned by the augmented solve is corrected for event-time shift at the `y=0` hit before evaluating the sensitivity of `x^2`

If `S = \partial_\varepsilon X` at the event time and `f` is the flow, then with `g(X)=y`,

\[
\tau_\varepsilon = - \frac{S_y}{f_y},
\qquad
\partial_\varepsilon X_{\mathrm{event}} = S + f\,\tau_\varepsilon.
\]

The saved scalar is then

\[
\partial_\varepsilon(x^2) = 2x\,(\partial_\varepsilon X_{\mathrm{event}})_x.
\]

## No repeat work across iterates

Successive iterates do **not** trigger new solves.

Each grid point is handled by:

1. one forward-sensitivity integration
2. one pass through the trajectory
3. one event list containing up to `ATTEMPT031_MAX_EVENT_ITERATES` `|x|`-maxima

The plotter later reads those saved per-iterate sensitivities and contours whichever iterate indices are requested.

## Output schema

Each saved row contains:

- `alpha`
- `lambda`
- `absxmax_count`
- `absxmax_sensitivity_values`
- `absxmax_return_times`
- `absxmax_states`
- `status`

Notes:

- `absxmax_sensitivity_values` are signed real sensitivities, not signs only
- heatmaps should use `abs.(absxmax_sensitivity_values)`
- `absxmax_return_times` are the refined event times on the exact `y=0` section

## Default run

The default full run is:

- grid: `500 x 500`
- stored iterates: `16`
- contour overlay: iterates `2:7`

Use the provided runner:

- [run_grid500_branch16_zsensabsx_plot7_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-031/run_grid500_branch16_zsensabsx_plot7_upload.sh)

## Main files

- [main.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-031/main.jl)
- [contours.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-031/contours.jl)
- [run_grid500_branch16_zsensabsx_plot7_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-031/run_grid500_branch16_zsensabsx_plot7_upload.sh)
