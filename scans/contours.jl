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

function edge_zero_point(
  edge_id::Int,
  values::NTuple{4, Float64},
  x_tl::Float64,
  y_tl::Float64,
  x_tr::Float64,
  y_tr::Float64,
  x_br::Float64,
  y_br::Float64,
  x_bl::Float64,
  y_bl::Float64,
  level::Float64,
)::Union{Nothing, Tuple{Float64, Float64}}
  z_tl, z_tr, z_br, z_bl = values
  if edge_id == 1
    z1, z2 = z_tl, z_tr
    x1, y1, x2, y2 = x_tl, y_tl, x_tr, y_tr
  elseif edge_id == 2
    z1, z2 = z_tr, z_br
    x1, y1, x2, y2 = x_tr, y_tr, x_br, y_br
  elseif edge_id == 3
    z1, z2 = z_br, z_bl
    x1, y1, x2, y2 = x_br, y_br, x_bl, y_bl
  else
    z1, z2 = z_bl, z_tl
    x1, y1, x2, y2 = x_bl, y_bl, x_tl, y_tl
  end

  d1 = z1 - level
  d2 = z2 - level
  if !isfinite(d1) || !isfinite(d2) || (d1 == 0.0 && d2 == 0.0)
    return nothing
  elseif d1 == 0.0
    theta = 0.0
  elseif d2 == 0.0
    theta = 1.0
  elseif signbit(d1) == signbit(d2)
    return nothing
  else
    theta = d1 / (d1 - d2)
  end

  return ((1.0 - theta) * x1 + theta * x2, (1.0 - theta) * y1 + theta * y2)
end

function march_squares_zero_segments(
  Z::AbstractMatrix{<:Real},
  x_vals::AbstractVector{<:Real},
  y_vals::AbstractVector{<:Real};
  level::Float64=0.0,
)::Vector{NTuple{4, Float64}}
  segments = NTuple{4, Float64}[]

  for j in 1:size(Z, 1)-1
    for i in 1:size(Z, 2)-1
      x_tl = Float64(x_vals[i]); y_tl = Float64(y_vals[j])
      x_tr = Float64(x_vals[i + 1]); y_tr = Float64(y_vals[j])
      x_br = Float64(x_vals[i + 1]); y_br = Float64(y_vals[j + 1])
      x_bl = Float64(x_vals[i]); y_bl = Float64(y_vals[j + 1])

      z_tl = Float64(Z[j, i])
      z_tr = Float64(Z[j, i + 1])
      z_br = Float64(Z[j + 1, i + 1])
      z_bl = Float64(Z[j + 1, i])
      all(isfinite, (z_tl, z_tr, z_br, z_bl)) || continue

      case_idx =
        (z_tl >= level ? 8 : 0) +
        (z_tr >= level ? 4 : 0) +
        (z_br >= level ? 2 : 0) +
        (z_bl >= level ? 1 : 0)
      case_idx == 0 && continue
      case_idx == 15 && continue

      values = (z_tl, z_tr, z_br, z_bl)
      points = Dict{Int, Tuple{Float64, Float64}}()
      for edge_id in 1:4
        point = edge_zero_point(edge_id, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
        isnothing(point) || (points[edge_id] = point)
      end

      if case_idx == 5 || case_idx == 10
        center_value = 0.25 * (z_tl + z_tr + z_br + z_bl)
        pairing =
          case_idx == 5 ?
          (center_value >= level ? ((1, 2), (3, 4)) : ((1, 4), (2, 3))) :
          (center_value >= level ? ((1, 4), (2, 3)) : ((1, 2), (3, 4)))
        for (edge_a, edge_b) in pairing
          haskey(points, edge_a) && haskey(points, edge_b) || continue
          x1, y1 = points[edge_a]
          x2, y2 = points[edge_b]
          push!(segments, (x1, y1, x2, y2))
        end
        continue
      end

      pairing =
        case_idx == 1 ? ((4, 3),) :
        case_idx == 2 ? ((3, 2),) :
        case_idx == 3 ? ((4, 2),) :
        case_idx == 4 ? ((1, 2),) :
        case_idx == 6 ? ((1, 3),) :
        case_idx == 7 ? ((1, 4),) :
        case_idx == 8 ? ((1, 4),) :
        case_idx == 9 ? ((1, 3),) :
        case_idx == 11 ? ((1, 2),) :
        case_idx == 12 ? ((4, 2),) :
        case_idx == 13 ? ((3, 2),) :
        case_idx == 14 ? ((4, 3),) :
        ()

      for (edge_a, edge_b) in pairing
        haskey(points, edge_a) && haskey(points, edge_b) || continue
        x1, y1 = points[edge_a]
        x2, y2 = points[edge_b]
        push!(segments, (x1, y1, x2, y2))
      end
    end
  end

  return segments
end

function segments_to_nan_polyline(
  segments::Vector{NTuple{4, Float64}}
)::Tuple{Vector{Float64}, Vector{Float64}}
  xs = Float64[]
  ys = Float64[]
  sizehint!(xs, 3 * length(segments))
  sizehint!(ys, 3 * length(segments))

  for (x1, y1, x2, y2) in segments
    append!(xs, (x1, x2, NaN))
    append!(ys, (y1, y2, NaN))
  end

  isempty(xs) && return xs, ys
  return xs[1:end-1], ys[1:end-1]
end
