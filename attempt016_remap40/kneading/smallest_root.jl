using PolynomialRoots

function smallest_root(p::Vector{Int})
  roots_sorted = sort(roots(p), by=abs)
  return abs(roots_sorted[1])
end

