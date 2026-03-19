# Affine modulo family on [0, 1]:
# f_{beta,alpha}(x) = beta * x + alpha mod 1
#
# Parameter convention:
# p[1] = beta
# p[2] = alpha
#
# For alpha reduced modulo 1 and beta > 0, the discontinuities occur at
# x = (k - alpha) / beta for integers k with alpha < k < alpha + beta.

const AFFINE_MODULO_LEFT_GERM = Int8(-1)
const AFFINE_MODULO_RIGHT_GERM = Int8(1)

struct AffineModuloGerm
  x::Float64
  side::Int8
end

@inline function affine_modulo_normalized_alpha(alpha)
  return mod(alpha, 1.0)
end

@inline function affine_modulo_parameters(p)
  beta = Float64(p[1])
  alpha = affine_modulo_normalized_alpha(Float64(p[2]))
  beta > 0 || throw(ArgumentError("affine modulo scan requires beta > 0"))
  return beta, alpha
end

function affine_modulo_map(p, x)
  beta, alpha = affine_modulo_parameters(p)
  return mod(beta * x + alpha, 1.0)
end

function affine_modulo_discontinuities(p)
  beta, alpha = affine_modulo_parameters(p)
  max_jump_index = floor(Int, prevfloat(alpha + beta))
  max_jump_index <= 0 && return Float64[]

  discontinuities = Vector{Float64}(undef, max_jump_index)
  for k in 1:max_jump_index
    discontinuities[k] = (k - alpha) / beta
  end
  return discontinuities
end

function affine_modulo_recommended_scan_rectangle()
  return (
    alpha=(0.0, 0.5),
    beta=(1.0, 1.5),
    regime="single-discontinuity expanding strip",
  )
end

@inline function affine_modulo_integer_tolerance(x)
  return 512 * eps(max(1.0, abs(x)))
end

@inline function affine_modulo_hits_integer(x)
  nearest_integer = round(x)
  return abs(x - nearest_integer) <= affine_modulo_integer_tolerance(x)
end

function affine_modulo_iterate_germ(germ::AffineModuloGerm, p)
  beta, alpha = affine_modulo_parameters(p)
  lifted_image = beta * germ.x + alpha

  if affine_modulo_hits_integer(lifted_image)
    if germ.side == AFFINE_MODULO_LEFT_GERM
      return AffineModuloGerm(1.0, AFFINE_MODULO_LEFT_GERM)
    elseif germ.side == AFFINE_MODULO_RIGHT_GERM
      return AffineModuloGerm(0.0, AFFINE_MODULO_RIGHT_GERM)
    end
  end

  return AffineModuloGerm(mod(lifted_image, 1.0), germ.side)
end

function affine_modulo_lap_index(germ::AffineModuloGerm, discontinuities::Vector{Float64})
  x = germ.x
  n_laps = length(discontinuities) + 1

  x <= 0.0 && return 1
  x >= 1.0 && return n_laps

  idx = searchsortedlast(discontinuities, x)
  if idx > 0 && abs(x - discontinuities[idx]) <= affine_modulo_integer_tolerance(x)
    return germ.side == AFFINE_MODULO_RIGHT_GERM ? idx + 1 : idx
  end

  return idx + 1
end

function allocate_affine_modulo_kneading_matrix(discontinuities, K)
  m = length(discontinuities)
  return zeros(Int8, m, m + 1, K + 1)
end

function affine_modulo_kneading_matrix!(
  matrix::Array{Int8, 3},
  discontinuities::Vector{Float64},
  p
)
  fill!(matrix, 0)
  m = length(discontinuities)
  K = size(matrix, 3) - 1

  @assert size(matrix, 1) == m
  @assert size(matrix, 2) == m + 1

  for i in 1:m
    # Milnor-Thurston kneading increment:
    # theta(c_i^+) - theta(c_i^-)
    matrix[i, i, 1] -= 1
    matrix[i, i + 1, 1] += 1

    left_germ = AffineModuloGerm(discontinuities[i], AFFINE_MODULO_LEFT_GERM)
    right_germ = AffineModuloGerm(discontinuities[i], AFFINE_MODULO_RIGHT_GERM)

    for k in 2:K + 1
      left_germ = affine_modulo_iterate_germ(left_germ, p)
      right_germ = affine_modulo_iterate_germ(right_germ, p)

      left_lap = affine_modulo_lap_index(left_germ, discontinuities)
      right_lap = affine_modulo_lap_index(right_germ, discontinuities)

      matrix[i, left_lap, k] -= 1
      matrix[i, right_lap, k] += 1
    end
  end

  return matrix
end

function affine_modulo_code_type(max_iterates::Int)
  max_code = big(3)^max_iterates - 1
  if max_code <= typemax(UInt32)
    return UInt32
  elseif max_code <= typemax(UInt64)
    return UInt64
  elseif max_code <= typemax(UInt128)
    return UInt128
  end

  throw(ArgumentError("affine modulo prefix codes exceed UInt128 capacity"))
end

@inline function affine_modulo_pair_state(::Type{T}, left_lap::Int, right_lap::Int) where {T <: Unsigned}
  if left_lap == right_lap
    return T(0)
  elseif left_lap < right_lap
    return T(1)
  else
    return T(2)
  end
end

function affine_modulo_exact_prefix_codes(p, max_iterates::Int, ::Type{T}=UInt128) where {T <: Unsigned}
  discontinuities = affine_modulo_discontinuities(p)
  @assert length(discontinuities) == 1 "exact prefix coding assumes exactly one discontinuity"

  left_germ = AffineModuloGerm(discontinuities[1], AFFINE_MODULO_LEFT_GERM)
  right_germ = AffineModuloGerm(discontinuities[1], AFFINE_MODULO_RIGHT_GERM)

  codes = Vector{T}(undef, max_iterates)
  code = T(0)

  for iterate in 1:max_iterates
    left_germ = affine_modulo_iterate_germ(left_germ, p)
    right_germ = affine_modulo_iterate_germ(right_germ, p)

    left_lap = affine_modulo_lap_index(left_germ, discontinuities)
    right_lap = affine_modulo_lap_index(right_germ, discontinuities)

    code = T(3) * code + affine_modulo_pair_state(T, left_lap, right_lap)
    codes[iterate] = code
  end

  return codes
end
