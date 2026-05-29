function add(
  x::Vector{Integer},
  y::Vector{Integer}
)::Vector{Integer}
  m = length(x)
  n = length(y)
  if m < n
    result = zeros(Integer, n)
    for i in 1:m
      result[i] = x[i] + y[i]
    end
    for i in m+1:n
      result[i] = y[i]
    end
  else
    result = zeros(Integer, m)
    for i in 1:n
      result[i] = x[i] + y[i]
    end
    for i in n+1:m
      result[i] = x[i]
    end
  end
  return result
end

function add!(
  x::Vector{Integer},
  y::Vector{Integer}
)
  @assert length(x) == length(y)
  n = length(x)
  for i in 1:n
    x[i] += y[i]
  end
end

function multiply(
  x::Vector{Integer},
  y::Vector{Integer}
)::Vector{Integer}
  m = length(x)
  n = length(y)
  # The result has length n + m - 1 if both arrays are
  # non-empty; if either is empty, the result should be
  # empty.
  if m == 0 || n == 0
    return Integer[]
  end
  
  result = zeros(Integer, m + n - 1)
  for i in 1:m
    for j in 1:n
      result[i + j - 1] += x[i] * y[j]
    end
  end
  return result
end

function scale(
  a::Number,
  x::Vector{Integer}
)::Vector{Integer}
  return [a * n for n in x]
end

