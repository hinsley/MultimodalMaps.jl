using Test

include("../maps/chebyshev_cubic.jl")
include("../maps/kneading/affine_modulo.jl")
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

    affine_p = (1.25, 0.20)
    affine_discontinuities = affine_modulo_discontinuities(affine_p)
    @test length(affine_discontinuities) == 1

    affine_left = AffineModuloGerm(affine_discontinuities[1], AFFINE_MODULO_LEFT_GERM)
    affine_right = AffineModuloGerm(affine_discontinuities[1], AFFINE_MODULO_RIGHT_GERM)
    affine_left_image = affine_modulo_iterate_germ(affine_left, affine_p)
    affine_right_image = affine_modulo_iterate_germ(affine_right, affine_p)

    @test affine_left_image == AffineModuloGerm(1.0, AFFINE_MODULO_LEFT_GERM)
    @test affine_right_image == AffineModuloGerm(0.0, AFFINE_MODULO_RIGHT_GERM)
    @test affine_modulo_lap_index(affine_left_image, affine_discontinuities) == 2
    @test affine_modulo_lap_index(affine_right_image, affine_discontinuities) == 1

    affine_matrix = allocate_affine_modulo_kneading_matrix(affine_discontinuities, 6)
    affine_modulo_kneading_matrix!(affine_matrix, affine_discontinuities, affine_p)
    @test vec(affine_matrix[1, :, 1]) == Int8[-1, 1]
    @test vec(affine_matrix[1, :, 2]) == Int8[1, -1]
    affine_codes = affine_modulo_exact_prefix_codes(affine_p, 6, UInt32)
    affine_code = UInt32(0)
    for iterate in 1:6
        slice = vec(affine_matrix[1, :, iterate + 1])
        digit = if slice[1] == 0 && slice[2] == 0
            UInt32(0)
        elseif slice[1] == -1 && slice[2] == 1
            UInt32(1)
        elseif slice[1] == 1 && slice[2] == -1
            UInt32(2)
        else
            error("unexpected affine modulo matrix slice $(Tuple(slice))")
        end

        affine_code = UInt32(3) * affine_code + digit
        @test affine_code == affine_codes[iterate]
    end

    affine_code_type = affine_modulo_code_type(20)
    @test affine_code_type == UInt32
    @test length(affine_modulo_discontinuities((nextfloat(1.0), nextfloat(0.0)))) == 1
end
