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

is_continuous = true
