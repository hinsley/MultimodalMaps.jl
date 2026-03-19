# Chebyshev cubic family:
# f_{u,v}(x) = ((u-v)/2) * (4x^3 - 3x) + (u+v)/2
# Parameters:
# p[1] = u = f(-1/2)
# p[2] = v = f(1/2)
function map(p, x)
  u = p[1]
  v = p[2]
  return ((u - v) / 2) * (4 * x^3 - 3 * x) + ((u + v) / 2)
end

function derivative(p, x)
  u = p[1]
  v = p[2]
  return ((u - v) / 2) * (12 * x^2 - 3)
end

function critical_points(p)
  return [-0.5, 0.5]
end

function chebyshev_cubic_lap(point, critical_points)
  lap = 1
  for j in 1:length(critical_points)
    if point >= critical_points[j]
      lap = j + 1
    else
      break
    end
  end
  return lap
end

function chebyshev_cubic_escape_lap(lap, orientation)
  @assert lap == 1 || lap == 3
  return orientation > 0 ? lap : (lap == 1 ? 3 : 1)
end

function chebyshev_cubic_kneading_matrix!(matrix::Array{Int8}, critical_points, p)
  orientation = map(p, critical_points[1]) > map(p, critical_points[2]) ? 1 : -1
  m = length(critical_points)
  K = size(matrix, 3) - 1

  for i in 1:m
    matrix[i, i,   1] =  1 * orientation * (isodd(i) ? 1 : -1)
    matrix[i, i+1, 1] = -1 * orientation * (isodd(i) ? 1 : -1)

    point = critical_points[i]
    cumulative_orientation = orientation
    escaped = false
    lap = i

    for k in 2:K+1
      if escaped
        lap = chebyshev_cubic_escape_lap(lap, orientation)
      else
        point = map(p, point)
        if isfinite(point)
          lap = chebyshev_cubic_lap(point, critical_points)
        elseif isinf(point)
          lap = point > 0 ? m + 1 : 1
          escaped = true
        else
          # A finite cubic iterate should not jump straight to NaN, but if it
          # does, keep following the escaped side instead of collapsing to lap 1.
          lap = chebyshev_cubic_escape_lap(lap, orientation)
          escaped = true
        end
      end

      cumulative_orientation *= orientation * (isodd(lap) ? 1 : -1)
      matrix[i, lap, k] += 2 * cumulative_orientation
    end
  end
end

is_continuous = true
