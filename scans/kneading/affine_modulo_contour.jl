using Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(REPO_ROOT)

if isempty(get(ENV, "DISPLAY", "")) && !haskey(ENV, "GKSwstype")
  ENV["GKSwstype"] = "100"
end

using Plots

include(joinpath(REPO_ROOT, "scans", "contours.jl"))
include(joinpath(REPO_ROOT, "maps", "kneading", "affine_modulo.jl"))

const OUTPUT_TAG = "CriticalOrbit"

function env_int(name::String, default::Int)::Int
  return parse(Int, get(ENV, name, string(default)))
end

function contour_alpha(iterate::Int)::Float64
  return Float64(iterate - 1)^(-1.2)
end

function affine_modulo_orbit_offsets(
  beta::Float64,
  alpha::Float64,
  iterate::Int,
)::Tuple{Float64, Float64}
  p = (beta, alpha)
  discontinuities = affine_modulo_discontinuities(p)
  @assert length(discontinuities) == 1
  discontinuity = discontinuities[1]

  zero_side_germ = AffineModuloGerm(discontinuity, AFFINE_MODULO_RIGHT_GERM)
  one_side_germ = AffineModuloGerm(discontinuity, AFFINE_MODULO_LEFT_GERM)

  for _ in 2:iterate
    zero_side_germ = affine_modulo_iterate_germ(zero_side_germ, p)
    one_side_germ = affine_modulo_iterate_germ(one_side_germ, p)
  end

  return zero_side_germ.x - discontinuity, one_side_germ.x - discontinuity
end

rectangle = affine_modulo_recommended_scan_rectangle()
alpha_min, alpha_max = rectangle.alpha
beta_min, beta_max = rectangle.beta

grid_size = env_int("AFFINE_MODULO_GRID_SIZE", 1000)
iterates = env_int("AFFINE_MODULO_ITERATES", 20)
frameless = lowercase(get(ENV, "AFFINE_MODULO_FRAMELESS", "false")) == "true"
contour_linewidth = 2.0

# Nudge sampling off the strip boundaries so the sampled grid stays in the
# one-discontinuity regime while the axes still show the natural rectangle.
alpha_vals = range(nextfloat(alpha_min), stop=prevfloat(alpha_max), length=grid_size)
beta_vals = range(nextfloat(beta_min), stop=prevfloat(beta_max), length=grid_size)

xticks = frameless ? nothing : collect(alpha_min:0.5:alpha_max)
yticks = frameless ? nothing : collect(beta_min:0.5:beta_max)
orbit_grids = [zeros(Float64, length(beta_vals), length(alpha_vals)) for _ in 1:2]

fig = plot(
  aspect_ratio=:equal,
  colorbar=false,
  xlims=(alpha_min, alpha_max),
  ylims=(beta_min, beta_max),
  xlabel=frameless ? "" : raw"$\alpha$",
  ylabel=frameless ? "" : raw"$\beta$",
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
  xticks=xticks,
  yticks=yticks,
  ticks=frameless ? nothing : :auto,
  background_color=:white,
)

compute_seconds = 0.0
contour_seconds = 0.0

for iterate in iterates:-1:2
  global compute_seconds += @elapsed begin
    for i in eachindex(alpha_vals)
      alpha = Float64(alpha_vals[i])
      for j in eachindex(beta_vals)
        beta = Float64(beta_vals[j])
        zero_side_offset, one_side_offset = affine_modulo_orbit_offsets(beta, alpha, iterate)
        orbit_grids[1][j, i] = zero_side_offset
        orbit_grids[2][j, i] = one_side_offset
      end
    end
  end

  alpha_value = contour_alpha(iterate)
  contour_specs = (
    (1, RGBA(0.85, 0.15, 0.12, alpha_value)),
    (2, RGBA(0.10, 0.30, 0.90, alpha_value)),
  )

  global contour_seconds += @elapsed begin
    for (grid_idx, contour_color) in contour_specs
      contour_segments = march_squares_zero_segments(
        orbit_grids[grid_idx],
        alpha_vals,
        beta_vals;
        level=0.0,
      )
      contour_xs, contour_ys = segments_to_nan_polyline(contour_segments)
      isempty(contour_xs) && continue
      plot!(
        fig,
        contour_xs,
        contour_ys,
        color=contour_color,
        linewidth=contour_linewidth,
        label=false,
      )
    end
  end
end

render_seconds = @elapsed begin
  if get(ENV, "GKSwstype", "") != "100"
    display(fig)
  end

  output_stem = joinpath(
    @__DIR__,
    "affine_modulo_kneading_diagram_$(OUTPUT_TAG)_$(iterates)",
  )
  if frameless
    savefig(fig, "$(output_stem)_frameless.png")
  else
    savefig(fig, "$(output_stem).png")
  end
end

runtime_path = joinpath(
  @__DIR__,
  "affine_modulo_kneading_diagram_$(OUTPUT_TAG)_$(iterates)_runtime.txt",
)
open(runtime_path, "w") do io
  println(io, "alpha_range\t[$alpha_min, $alpha_max]")
  println(io, "beta_range\t[$beta_min, $beta_max]")
  println(io, "grid_size\t$grid_size")
  println(io, "iterates\t$iterates")
  println(io, "compute_seconds\t$(round(compute_seconds; digits=6))")
  println(io, "contour_seconds\t$(round(contour_seconds; digits=6))")
  println(io, "render_seconds\t$(round(render_seconds; digits=6))")
  println(io, "regime\t$(rectangle.regime)")
end

println("Affine modulo compute runtime: $(round(compute_seconds; digits=6)) seconds")
println("Affine modulo contour extraction runtime: $(round(contour_seconds; digits=6)) seconds")
println("Affine modulo render/save runtime: $(round(render_seconds; digits=6)) seconds")
