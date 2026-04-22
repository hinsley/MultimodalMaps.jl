using Pkg
Pkg.activate(".")

if isempty(get(ENV, "DISPLAY", "")) && !haskey(ENV, "GKSwstype")
  ENV["GKSwstype"] = "100"
end

using Plots

include("../scans/contours.jl")
include("../maps/chebyshev_cubic.jl")

function env_int(name::String, default::Int)::Int
  return parse(Int, get(ENV, name, string(default)))
end

function contour_alpha(iterate::Int)::Float64
  return Float64(iterate - 1)^(-1.2)
end

@inline function chebyshev_map_uv(u::Float64, v::Float64, x::Float64)::Float64
  return ((u - v) / 2) * (4 * x^3 - 3 * x) + ((u + v) / 2)
end

function critical_orbit_value(
  u::Float64,
  v::Float64,
  start_point::Float64,
  iterate::Int,
)::Float64
  point = start_point
  for _ in 2:iterate
    point = chebyshev_map_uv(u, v, point)
    isfinite(point) || return point
  end
  return point
end

grid_length = env_int("CHEBYSHEV_GRID_LENGTH", 1000)
u_vals = range(-2.0, stop=2.0, length=grid_length)
v_vals = range(-2.0, stop=2.0, length=grid_length)

iterates = env_int("CHEBYSHEV_ITERATES", 20)
contour_linewidth = 1.0
frameless = false
compute_seconds = 0.0
axis_ticks = frameless ? nothing : collect(-2.0:0.5:2.0)
critical_boundaries = Float64.(critical_points([0.0, 0.0]))
orbit_grids = [zeros(Float64, length(v_vals), length(u_vals)) for _ in eachindex(critical_boundaries)]

fig = plot(
  aspect_ratio=:equal,
  colorbar=false,
  xlims=(minimum(u_vals), maximum(u_vals)),
  ylims=(minimum(v_vals), maximum(v_vals)),
  xlabel=frameless ? "" : raw"$u$",
  ylabel=frameless ? "" : raw"$v$",
  legend=false,
  size=(1000, 1000),
  xguidefontsize=30,
  yguidefontsize=30,
  xtickfontsize=16,
  ytickfontsize=16,
  left_margin=frameless ? -5Plots.mm : 5Plots.mm,
  bottom_margin=frameless ? -2.5Plots.mm : 5Plots.mm,
  right_margin=frameless ? -2Plots.mm : 3Plots.mm,
  top_margin=frameless ? -5Plots.mm : 3Plots.mm,
  framestyle=frameless ? :none : :auto,
  grid=frameless ? false : :auto,
  background_color=:white,
  ticks=axis_ticks
)
for iterate in iterates:-1:2
  global compute_seconds += @elapsed begin
    for i in 1:length(u_vals)
      u = Float64(u_vals[i])
      for j in 1:length(v_vals)
        v = Float64(v_vals[j])
        for critical_idx in eachindex(critical_boundaries)
          orbit_grids[critical_idx][j, i] = critical_orbit_value(
            u,
            v,
            critical_boundaries[critical_idx],
            iterate
          )
        end
      end
    end
  end

  alpha = contour_alpha(iterate)
  contour_specs = (
    (1, RGBA(0.85, 0.15, 0.12, alpha)),
    (2, RGBA(0.10, 0.30, 0.90, alpha))
  )
  for (critical_idx, contour_color) in contour_specs
    orbit_grid = orbit_grids[critical_idx]
    for boundary in critical_boundaries
      contour_segments = march_squares_zero_segments(
        orbit_grid,
        u_vals,
        v_vals;
        level=boundary
      )
      contour_xs, contour_ys = segments_to_nan_polyline(contour_segments)
      isempty(contour_xs) && continue
      plot!(
        fig,
        contour_xs,
        contour_ys,
        color=contour_color,
        linewidth=contour_linewidth,
        label=false
      )
    end
  end
end

if get(ENV, "GKSwstype", "") != "100"
  display(fig)
end

output_tag = "CriticalOrbit"
output_stem = "chebyshev_cubic_kneading_diagram_$(output_tag)_$(iterates)"
if frameless
  savefig(fig, "$(output_stem)_frameless.png")
else
  savefig(fig, "$(output_stem).png")
end

println("compute_seconds=$(compute_seconds)")
