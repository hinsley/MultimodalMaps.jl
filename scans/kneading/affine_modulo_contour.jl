using Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(REPO_ROOT)

if isempty(get(ENV, "DISPLAY", "")) && !haskey(ENV, "GKSwstype")
  ENV["GKSwstype"] = "100"
end

using Plots

include(joinpath(REPO_ROOT, "kneading", "encodings.jl"))
include(joinpath(REPO_ROOT, "scans", "contours.jl"))
include(joinpath(REPO_ROOT, "maps", "kneading", "affine_modulo.jl"))

const OUTPUT_TAG = "ExactMatrix"

function affine_modulo_plot_color(iterate, max_iterates, color_exp)
  shade = ((iterate - 2) / max_iterates)^(1 / color_exp)
  return RGB(shade, shade, shade)
end

function affine_modulo_validate_prefix_coding(max_iterates)
  code_type = affine_modulo_code_type(max_iterates)
  sample_parameters = (
    (nextfloat(1.0), nextfloat(0.0)),
    (1.10, 0.10),
    (1.25, 0.20),
    (1.40, 0.35),
  )

  for p in sample_parameters
    discontinuities = affine_modulo_discontinuities(p)
    @assert length(discontinuities) == 1

    matrix = allocate_affine_modulo_kneading_matrix(discontinuities, max_iterates)
    affine_modulo_kneading_matrix!(matrix, discontinuities, p)
    codes = affine_modulo_exact_prefix_codes(p, max_iterates, code_type)

    code = code_type(0)
    for iterate in 1:max_iterates
      slice = vec(matrix[1, :, iterate + 1])
      if slice[1] == 0 && slice[2] == 0
        digit = code_type(0)
      elseif slice[1] == -1 && slice[2] == 1
        digit = code_type(1)
      elseif slice[1] == 1 && slice[2] == -1
        digit = code_type(2)
      else
        error("unexpected affine modulo matrix slice $(Tuple(slice))")
      end

      code = code_type(3) * code + digit
      @assert code == codes[iterate]
    end
  end
end

rectangle = affine_modulo_recommended_scan_rectangle()
alpha_min, alpha_max = rectangle.alpha
beta_min, beta_max = rectangle.beta

grid_size = parse(Int, get(ENV, "AFFINE_MODULO_GRID_SIZE", "1000"))
iterates = parse(Int, get(ENV, "AFFINE_MODULO_ITERATES", "20"))
color_exp = parse(Float64, get(ENV, "AFFINE_MODULO_COLOR_EXP", "2.0"))
frameless = lowercase(get(ENV, "AFFINE_MODULO_FRAMELESS", "false")) == "true"

# Nudge sampling off the strip boundaries so the entire sampled grid stays in the
# intended one-discontinuity regime, while the plot still shows the natural
# rectangle [0, 0.5] x [1, 1.5].
alpha_vals = range(nextfloat(alpha_min), stop=prevfloat(alpha_max), length=grid_size)
beta_vals = range(nextfloat(beta_min), stop=prevfloat(beta_max), length=grid_size)

xticks = frameless ? nothing : collect(alpha_min:0.5:alpha_max)
yticks = frameless ? nothing : collect(beta_min:0.5:beta_max)

affine_modulo_validate_prefix_coding(iterates)

label_grids = [zeros(UInt32, length(beta_vals), length(alpha_vals)) for _ in 2:iterates]
label_lookups = [Dict{NTuple{6, UInt64}, UInt32}() for _ in 2:iterates]
next_labels = [Ref(UInt32(1)) for _ in 2:iterates]
contours = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, iterates - 1)
contour_seconds = 0.0

compute_seconds = @elapsed begin
  for (i, alpha) in enumerate(alpha_vals)
    for (j, beta) in enumerate(beta_vals)
      p = (beta, alpha)
      discontinuities = affine_modulo_discontinuities(p)
      @assert length(discontinuities) == 1

      matrix = allocate_affine_modulo_kneading_matrix(discontinuities, iterates)
      affine_modulo_kneading_matrix!(matrix, discontinuities, p)

      for (grid_idx, iterate) in enumerate(2:iterates)
        label_grids[grid_idx][j, i] = exact_matrix_label!(
          label_lookups[grid_idx],
          next_labels[grid_idx],
          view(matrix, :, :, 1:iterate + 1),
        )
      end
    end
  end
end

contour_seconds = @elapsed begin
  for iterate in 2:iterates
    contours[iterate - 1] = march_squares_simple(
      label_grids[iterate - 1],
      alpha_vals,
      beta_vals,
    )
  end
end

render_seconds = @elapsed begin
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
  )

  for iterate in iterates:-1:2
    contour_xs, contour_ys = contours[iterate - 1]
    plot!(
      fig,
      contour_xs,
      contour_ys,
      color=affine_modulo_plot_color(iterate, iterates, color_exp),
      linewidth=1,
      label="Iterate $iterate",
    )
  end

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

println("Affine modulo compute-only runtime: $(round(compute_seconds; digits=6)) seconds")
println("Affine modulo contour extraction runtime: $(round(contour_seconds; digits=6)) seconds")
println("Affine modulo render/save runtime: $(round(render_seconds; digits=6)) seconds")
