function march_squares_simple(
  Z::AbstractMatrix,
  x_vals::AbstractVector{<:Real},
  y_vals::AbstractVector{<:Real}
)::Tuple{Vector{Float64}, Vector{Float64}}
  """
  March over the matrix Z, and return the x and y coordinates of the
  contours.
  Splits line segments via NaN values instead of returning
  multiple line segments to make plotting more efficient.
  """

  contour_xs = Float64[]
  contour_ys = Float64[]

  for j in 1:size(Z, 1)-1
    for i in 1:size(Z, 2)-1
      x_tl, y_tl = x_vals[i], y_vals[j]
      x_tr, y_tr = x_vals[i+1], y_vals[j]
      x_br, y_br = x_vals[i+1], y_vals[j+1]
      x_bl, y_bl = x_vals[i], y_vals[j+1]

      x_tm, y_tm = (x_tl + x_tr) / 2, (y_tl + y_tr) / 2
      x_rm, y_rm = (x_tr + x_br) / 2, (y_tr + y_br) / 2
      x_bm, y_bm = (x_br + x_bl) / 2, (y_br + y_bl) / 2
      x_lm, y_lm = (x_bl + x_tl) / 2, (y_bl + y_tl) / 2

      z_tl = Z[j, i]
      z_tr = Z[j, i+1]
      z_br = Z[j+1, i+1]
      z_bl = Z[j+1, i]

      if z_tl != z_tr && z_tr == z_br && z_br == z_bl
        append!(contour_xs, [x_tm, x_lm, NaN])
        append!(contour_ys, [y_tm, y_lm, NaN])
      elseif z_tr != z_tl && z_tl == z_br && z_br == z_bl
        append!(contour_xs, [x_rm, x_tm, NaN])
        append!(contour_ys, [y_rm, y_tm, NaN])
      elseif z_br != z_tr && z_tr == z_tl && z_tl == z_bl
        append!(contour_xs, [x_bm, x_rm, NaN])
        append!(contour_ys, [y_bm, y_rm, NaN])
      elseif z_bl != z_br && z_br == z_tr && z_tr == z_tl
        append!(contour_xs, [x_lm, x_bm, NaN])
        append!(contour_ys, [y_lm, y_bm, NaN])
      elseif z_tl == z_tr && z_bl == z_br && z_tl != z_bl
        append!(contour_xs, [x_lm, x_rm, NaN])
        append!(contour_ys, [y_lm, y_rm, NaN])
      elseif z_tl == z_bl && z_tr == z_br && z_tl != z_tr
        append!(contour_xs, [x_tm, x_bm, NaN])
        append!(contour_ys, [y_tm, y_bm, NaN])
      end
    end
  end

  return contour_xs[1:end-1], contour_ys[1:end-1]
end
