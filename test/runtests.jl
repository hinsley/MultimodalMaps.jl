using Test

include("../maps/chebyshev_cubic.jl")
include("../kneading/matrix.jl")
include("../kneading/encodings.jl")

@testset "MultimodalMaps" begin
    function build_matrix(p, K; chebyshev_escape_fix=false)
        crit_points = critical_points(p)
        matrix = allocate_kneading_matrix(crit_points, K)
        if chebyshev_escape_fix
            chebyshev_cubic_kneading_matrix!(matrix, crit_points, p)
        else
            kneading_matrix!(matrix, map, crit_points, p)
        end
        return matrix
    end

    p = [-1.3, 0.8]
    q = [-1.3, 0.8]
    r = [-0.8, 1.3]

    matrix_p = build_matrix(p, 6)
    matrix_q = build_matrix(q, 6)
    matrix_r = build_matrix(r, 6)

    @test exact_matrix_key(matrix_p) == exact_matrix_key(matrix_q)
    @test exact_matrix_key(matrix_p) != exact_matrix_key(matrix_r)
    @test exact_matrix_key(matrix_p) == exact_matrix_key(build_matrix(p, 6; chebyshev_escape_fix=true))

    escape_p = [1.2832832832832832, -2.0]
    escape_q = [2.0, -1.2832832832832832]
    @test exact_matrix_key(build_matrix(escape_p, 20)) != exact_matrix_key(build_matrix(escape_q, 20))
    @test exact_matrix_key(build_matrix(escape_p, 20; chebyshev_escape_fix=true)) == exact_matrix_key(build_matrix(escape_q, 20; chebyshev_escape_fix=true))

    alternating_p = [-2.0, 1.2832832832832832]
    alternating_q = [-1.2832832832832832, 2.0]
    @test exact_matrix_key(build_matrix(alternating_p, 20; chebyshev_escape_fix=true)) == exact_matrix_key(build_matrix(alternating_q, 20; chebyshev_escape_fix=true))
end
