using Test

include("../maps/chebyshev_cubic.jl")
include("../kneading/matrix.jl")
include("../kneading/encodings.jl")

@testset "MultimodalMaps" begin
    K = 6

    p = [-1.3, 0.8]
    q = [-1.3, 0.8]
    r = [-0.8, 1.3]

    crit_points_p = critical_points(p)
    matrix_p = allocate_kneading_matrix(crit_points_p, K)
    kneading_matrix!(matrix_p, map, crit_points_p, p)

    crit_points_q = critical_points(q)
    matrix_q = allocate_kneading_matrix(crit_points_q, K)
    kneading_matrix!(matrix_q, map, crit_points_q, q)

    crit_points_r = critical_points(r)
    matrix_r = allocate_kneading_matrix(crit_points_r, K)
    kneading_matrix!(matrix_r, map, crit_points_r, r)

    @test exact_matrix_key(matrix_p) == exact_matrix_key(matrix_q)
    @test exact_matrix_key(matrix_p) != exact_matrix_key(matrix_r)
end
