# Attempt 032

This attempt uses forward sensitivities to build the one-step Jacobian of the 2D `|x|`-max return map on the `y=0` section, but it does **not** contour an ambient coordinate partial. Instead it contracts the one-step Jacobian with the sampled section-filament tangent obtained directly from a separate seed forward-sensitivity solve.

## Goal

For each parameter point `(alpha, lambda)` in the Shimizu-Morioka sweep:

- seed one unstable-manifold branch of the origin equilibrium exactly as before
- integrate first to the **first actual** local `|x|`-maximum on the attractor
- obtain the section-filament tangent at every sampled `|x|`-maximum directly from one seed forward-sensitivity solve from the unstable-manifold seed
- then, for each current `|x|`-maximum in the return sequence, restart a two-parameter forward-sensitivity solve from that current return state only
- integrate each restarted solve only to the **next** local `|x|`-maximum
- save the signed forward sensitivity
  \[
  \nabla_{(x_n,z_n)}(x_{n+1}^2)\cdot t_n
  \]
  where `t_n` is the propagated unit tangent of the 1D return-map domain filament in the `y=0` section

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

The contoured quantity is no longer the fixed-`z` ambient partial `\partial(x_{n+1}^2)/\partial x_n`. That partial produced too many transverse-coordinate zeroes. Instead this attempt measures the one-step derivative along the actual section filament.

The construction is:

- first, a one-parameter forward-sensitivity solve varies the unstable-manifold seed amplitude and is integrated through the whole sampled return sequence
- at each kept `|x|`-maximum, the event-corrected seed sensitivity gives a tangent vector of the sampled section filament in the `(x,z)` plane
- these are normalized pointwise to unit section tangents `t_n`
- next, at each current return state `X_n`, a two-parameter forward-sensitivity solve varies
  \[
  X_n(\varepsilon_x,\varepsilon_z)=X_{\mathrm{current\ |x|-max}} + (\varepsilon_x,0,\varepsilon_z)
  \]
- `SciMLSensitivity.ODEForwardSensitivityProblem` is used with the two section parameters `(\varepsilon_x,\varepsilon_z)`
- the forward sensitivities returned by the augmented solve are corrected for event-time shift at the next `y=0` hit
- these corrected sensitivities define the one-step section Jacobian
  \[
  D P_n =
  \begin{bmatrix}
  \partial x_{n+1}/\partial x_n & \partial x_{n+1}/\partial z_n\\
  \partial z_{n+1}/\partial x_n & \partial z_{n+1}/\partial z_n
  \end{bmatrix}
  \]
- the saved scalar is
  \[
  \nabla(x_{n+1}^2)\cdot t_n = 2x_{n+1}\,(DP_n t_n)_x
  \]
- the image-direction consistency is checked by comparing `DP_n t_n` to the independently measured next tangent `t_{n+1}`
- if the image tangent and the measured next tangent disagree too much, that iterate value is discarded as `NaN` rather than contoured

This means the tiny unstable-manifold seed near the origin is used only to sample the return-map filament and its tangents. The saved scalars are then one-step directional derivatives along that sampled filament, not ambient coordinate partials and not manually transported tangents.

If `S = \partial_\varepsilon X` at the event time and `f` is the flow, then with `g(X)=y`,

\[
\tau_\varepsilon = - \frac{S_y}{f_y},
\qquad
\partial_\varepsilon X_{\mathrm{event}} = S + f\,\tau_\varepsilon.
\]

For each one-parameter or two-parameter event solve, if `S` is a raw sensitivity vector at the event time and `f` is the flow, then with `g(X)=y`,

\[
\tau_\varepsilon = - \frac{S_y}{f_y},
\qquad
(\partial_\varepsilon X)_{\mathrm{event}} = S + f\,\tau_\varepsilon.
\]

The final saved scalar for iterate `n` is then

\[
\nabla(x_{n+1}^2)\cdot t_n = 2x_{n+1}\,(DP_n t_n)_x.
\]

## Per-iterate restarts

Successive iterates do trigger new short solves, because each one-step Jacobian `DP_n` depends on the current return state.

Each grid point is handled by:

1. one seed forward-sensitivity integration from the unstable-manifold seed to collect the sampled return states and their tangents
2. one two-parameter forward-sensitivity integration from the first `|x|`-maximum to the second
3. one two-parameter forward-sensitivity integration from the second `|x|`-maximum to the third
4. and so on, up to `ATTEMPT031_MAX_EVENT_ITERATES` one-step return segments

The plotter later reads those saved per-iterate directional sensitivities and contours whichever iterate indices are requested.

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

- `absxmax_sensitivity_values[k]` is the signed one-step directional derivative
  \[
  \nabla(x_{k+1}^2)\cdot t_k
  \]
  along the independently sampled unit tangent `t_k` of the current-return filament
- heatmaps should use `abs.(absxmax_sensitivity_values)`
- `absxmax_return_times[k]` is the refined one-step return time from the current `|x|`-maximum to the next one

## Default run

The default full run is:

- grid: `500 x 500`
- stored iterates: `16`
- contour overlay: iterates `2:7`

Use the provided runner:

- [run_grid500_branch16_xsensabsx_plot7_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-031/run_grid500_branch16_xsensabsx_plot7_upload.sh)
  in this attempt the runner is [run_grid500_branch16_filamentjac_plot7_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-032/run_grid500_branch16_filamentjac_plot7_upload.sh)

## Main files

- [main.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-032/main.jl)
- [contours.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-032/contours.jl)
- [run_grid500_branch16_filamentjac_plot7_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-032/run_grid500_branch16_filamentjac_plot7_upload.sh)
