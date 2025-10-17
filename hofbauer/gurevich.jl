# Gurevich entropy for the Hofbauer-Keller map.
# This allows us to approximate topological entropies of 1D unimodal maps from
# their Hofbauer-Keller kneading maps.

using Pkg
Pkg.activate(".")
Pkg.instantiate()
using SparseArrays
using LinearAlgebra
using Arpack

# Kneading map. Change this to change the function being considered.
function Q(k) # k starts at 1.
    # return max(k - 2, 0) # Fibonacci map.
    # return max(Int(floor(k/2)) - 1, 0) # Doubling map.
    # Map with period-n critical point ordered (c2, c3, ..., cn, c, c1).
    n = 44
    return k <= n-1 ? 0 : n-1
    # # Map with period-n critical point ordered (c2, c3, ..., cn-1, c, cn, c1).
    # n = 44
    # return k < n-1 ? max(0, k-n+3) : n-2
end

function cutting_times(max_n)
    Sk = [1, 2] # k starts at 0.
    k = 2
    while true
        # Use the rule S_k = S_{Q(k)} + S_{k-1}.
        Sk_next = Sk[Q(k)+1] + Sk[k] # Sk[i] is actually S_{i-1} because of 0-indexing.
        if Sk_next > max_n
            break
        end
        push!(Sk, Sk_next)
        k += 1
    end
    return Sk
end

function markov_edges(max_n, Sk)
    edges = [(n, n+1) for n in 1:max_n-1] # Trivial edges.

    # Descending edges.
    # Rule: D_{S_k} -> D_{S_{Q(k)}+1}.
    for k in 1:length(Sk)-1 # Recall k starts at 0, so max k is length(Sk)-1.
        push!(edges, (Sk[k+1], Sk[Q(k)+1]+1)) # Sk is 0-indexed, so add 1 to all indices.
    end

    return edges
end

# Construct a sparse adjacency matrix for the Markov chain.
function sparse_markov_matrix(max_n, Sk)
    edges = markov_edges(max_n, Sk)
    matrix = spzeros(max_n, max_n)
    for edge in edges
        matrix[edge[1], edge[2]] = 1
    end
    return matrix
end

# Compute the spectral radius of a (sparse) 0-1 Markov matrix efficiently.
function spectral_radius(A::AbstractMatrix{<:Real})
    if A isa SparseMatrixCSC
        if issymmetric(A)
            vals = eigs(Symmetric(A); nev=1, which=:LM)[1]
        else
            vals = eigs(A; nev=1, which=:LM)[1]
        end
    else
        SA = sparse(A)
        if issymmetric(SA)
            vals = eigs(Symmetric(SA); nev=1, which=:LM)[1]
        else
            vals = eigs(SA; nev=1, which=:LM)[1]
        end
    end
    return maximum(abs.(vals))
end

# Compute the Gurevich entropy of a Markov matrix from the spectral radius.
function gurevich_entropy(matrix)
    return log(spectral_radius(matrix))
end

# Example usage.
matrix = nothing
for max_n in 10_000:10_000
    Sk = cutting_times(max_n)
    matrix = sparse_markov_matrix(max_n, Sk)
    try
        h = gurevich_entropy(matrix)
        println("Gurevich entropy (max_n=$(max_n)): $(h)")
        println("exp(h) = $(exp(h))")
    catch err
        println("Error for max_n=$(max_n): $(err)")
        continue
    end
end
display(matrix)