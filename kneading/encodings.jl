function coefficient_encoding(coeff)
  return abs(coeff) * 2 - (coeff < 0 ? 1 : 0)
end

function exact_matrix_key(matrix)
  @assert length(matrix) <= 126 "exact_matrix_key supports matrices with up to 126 coefficients"

  chunk1 = UInt64(0)
  chunk2 = UInt64(0)
  chunk3 = UInt64(0)
  chunk4 = UInt64(0)
  chunk5 = UInt64(0)
  chunk6 = UInt64(0)

  bit_idx = 0
  for coeff in matrix
    digit = UInt64(coefficient_encoding(coeff))
    chunk_idx = bit_idx ÷ 64 + 1
    offset = bit_idx % 64
    lower = digit << offset
    upper = offset <= 61 ? UInt64(0) : digit >> (64 - offset)

    if chunk_idx == 1
      chunk1 |= lower
      chunk2 |= upper
    elseif chunk_idx == 2
      chunk2 |= lower
      chunk3 |= upper
    elseif chunk_idx == 3
      chunk3 |= lower
      chunk4 |= upper
    elseif chunk_idx == 4
      chunk4 |= lower
      chunk5 |= upper
    elseif chunk_idx == 5
      chunk5 |= lower
      chunk6 |= upper
    else
      chunk6 |= lower
    end

    bit_idx += 3
  end

  return (chunk1, chunk2, chunk3, chunk4, chunk5, chunk6)
end

function exact_matrix_label!(
  labels::Dict{NTuple{6, UInt64}, UInt32},
  next_label::Base.RefValue{UInt32},
  matrix
)::UInt32
  key = exact_matrix_key(matrix)
  return get!(labels, key) do
    label = next_label[]
    next_label[] = next_label[] + UInt32(1)
    label
  end
end

function determinant_encoding(det)
  K = length(det) - 1
  coeffs_encoded = Vector{Integer}(undef, K + 1)
  for k in 1:K+1
    coeffs_encoded[k] = coefficient_encoding(det[k])
  end
  alphabet_size = maximum(coeffs_encoded) + 1
  
  encoding = 0
  power = 1
  for k in 1:K+1
    encoding += coeffs_encoded[k] / alphabet_size^power
    power += 1
  end

  return encoding
end

function matrix_encoding(matrix)
  m = size(matrix, 1)
  n = size(matrix, 2)
  K = size(matrix, 3) - 1

  encoding = 0
  power = 1
  for k in 1:K+1
    for i in 1:m
      for j in 1:n
        encoding += coefficient_encoding(matrix[i, j, k]) * 5^power
        power += 1
      end
      power += 1
    end
    power += 1
  end

  return encoding
end
