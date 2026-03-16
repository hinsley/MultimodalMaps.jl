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

# Define parameter ranges.
u_vals = range(-1.0, stop=1.0, length=1000)
v_vals = range(-1.0, stop=1.0, length=1000)

# Allocate a matrix to store encoding values for each (u, v) pair.
Z = zeros(length(u_vals), length(v_vals))

# Choose the scan type: :matrix or :determinant.
scan_type = :matrix

# Choose the number of iterates.
iterates = 20

# Choose the color exponent for separation of iterates.
color_exp = 2

# Choose whether to save the figure without a frame.
frameless = false

# Calculate kneading diagram.
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
  @time for i in 1:length(u_vals)
    for j in 1:length(v_vals)
      u = u_vals[i]
      v = v_vals[j]
      p = [u, v]

      # Compute kneading matrix.
      crit_points = critical_points(p)
      matrix = allocate_kneading_matrix(crit_points, iterate)
      kneading_matrix!(matrix, map, crit_points, p)

      if scan_type == :matrix
        encoding = matrix_encoding(matrix)
      elseif scan_type == :determinant
        matrix = convert(Array{Integer, 3}, matrix)
        det = determinant(matrix[:, 2:end, :], false)
        encoding = determinant_encoding(det)
      end
      Z[j, i] = encoding
    end
  end

  # Create a contour plot.
  contour_xs, contour_ys = march_squares_simple(
    Z,
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

# Add a title to the plot.
if !frameless
  title!(
    fig,
    "Chebyshev cubic kneading diagram: $(uppercasefirst(string(scan_type))) encoding"
  )
end

# Display the plot when a GUI backend is available.
if get(ENV, "GKSwstype", "") != "100"
  display(fig)
end

# Save the plot to a file.
output_stem = "chebyshev_cubic_kneading_diagram_$(uppercasefirst(string(scan_type)))_$(iterates)"
if frameless
  savefig(fig, "$(output_stem)_frameless.png")
else
  savefig(fig, "$(output_stem).png")
end
