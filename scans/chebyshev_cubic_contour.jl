using Pkg
Pkg.activate(".")

if isempty(get(ENV, "DISPLAY", "")) && !haskey(ENV, "GKSwstype")
  ENV["GKSwstype"] = "100"
end

using Plots

include("../kneading/encodings.jl")
include("../kneading/matrix.jl")
include("../kneading/determinant.jl")
include("../scans/contours.jl")
include("../maps/chebyshev_cubic.jl")

u_vals = range(-2.0, stop=2.0, length=1000)
v_vals = range(-2.0, stop=2.0, length=1000)

numeric_values = zeros(Float64, length(v_vals), length(u_vals))
exact_labels = zeros(UInt32, length(v_vals), length(u_vals))

scan_type = :exact_matrix
iterates = 20
color_exp = 2
frameless = false

fig = plot(
  aspect_ratio=:equal,
  colorbar=false,
  xlims=(minimum(u_vals), maximum(u_vals)),
  ylims=(minimum(v_vals), maximum(v_vals)),
  xlabel=frameless ? "" : "u",
  ylabel=frameless ? "" : "v",
  legend=false,
  size=(1000, 1000),
  xguidefontsize=14,
  yguidefontsize=14,
  xtickfontsize=12,
  ytickfontsize=12,
  left_margin=frameless ? -5Plots.mm : 3Plots.mm,
  bottom_margin=frameless ? -2.5Plots.mm : 3Plots.mm,
  right_margin=frameless ? -2Plots.mm : 3Plots.mm,
  top_margin=frameless ? -5Plots.mm : 3Plots.mm,
  framestyle=frameless ? :none : :auto,
  grid=frameless ? false : :auto,
  ticks=frameless ? nothing : :auto
)
for iterate in iterates:-1:2
  label_lookup = Dict{NTuple{6, UInt64}, UInt32}()
  next_label = Ref(UInt32(1))

  @time for i in 1:length(u_vals)
    for j in 1:length(v_vals)
      u = u_vals[i]
      v = v_vals[j]
      p = [u, v]

      crit_points = critical_points(p)
      matrix = allocate_kneading_matrix(crit_points, iterate)
      chebyshev_cubic_kneading_matrix!(matrix, crit_points, p)

      if scan_type == :matrix
        numeric_values[j, i] = matrix_encoding(matrix)
      elseif scan_type == :exact_matrix
        exact_labels[j, i] = exact_matrix_label!(label_lookup, next_label, matrix)
      elseif scan_type == :determinant
        matrix_int = convert(Array{Integer, 3}, matrix)
        det = determinant(matrix_int[:, 2:end, :], false)
        numeric_values[j, i] = determinant_encoding(det)
      end
    end
  end

  contour_source = scan_type == :exact_matrix ? exact_labels : numeric_values
  contour_xs, contour_ys = march_squares_simple(
    contour_source,
    u_vals,
    v_vals
  )
  plot!(
    fig,
    contour_xs,
    contour_ys,
    color=RGB(
      ((iterate - 2) / iterates)^(1 / color_exp),
      ((iterate - 2) / iterates)^(1 / color_exp),
      ((iterate - 2) / iterates)^(1 / color_exp)
    ),
    linewidth=1,
    label="Iterate $iterate"
  )
end

scan_title = scan_type == :exact_matrix ? "Exact matrix labels" : "$(uppercasefirst(string(scan_type))) encoding"
if !frameless
  title!(fig, "Chebyshev cubic kneading diagram: $(scan_title)")
end

if get(ENV, "GKSwstype", "") != "100"
  display(fig)
end

output_tag = scan_type == :exact_matrix ? "ExactMatrix" : uppercasefirst(string(scan_type))
output_stem = "chebyshev_cubic_kneading_diagram_$(output_tag)_$(iterates)"
if frameless
  savefig(fig, "$(output_stem)_frameless.png")
else
  savefig(fig, "$(output_stem).png")
end
