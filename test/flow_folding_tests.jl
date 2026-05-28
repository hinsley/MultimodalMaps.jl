include("../flow_folding/FlowFolding.jl")

using .FlowFolding

@testset "flow_folding extrema and tangent signs" begin
    oscillator(u, p, t) = [u[2], -u[1]]
    oscillator_jacobian(u, p, t) = [0.0 1.0; -1.0 0.0]
    minima = FlowFoldingProblem(
        oscillator,
        nothing;
        variable_index=2,
        extremum=StateMinimum,
        dimension=2,
        jacobian=oscillator_jacobian,
    )
    maxima = FlowFoldingProblem(
        oscillator,
        nothing;
        variable_index=2,
        extremum=StateMaximum,
        dimension=2,
        jacobian=oscillator_jacobian,
    )

    @test accepts_extremum(minima, [0.0, -1.0]; event_atol=1e-12)
    @test !accepts_extremum(minima, [0.0, 1.0]; event_atol=1e-12)
    @test accepts_extremum(maxima, [0.0, 1.0]; event_atol=1e-12)

    min_events = collect_extrema(minima, [1.0, 0.0]; tspan=(0.0, 20.0), max_events=3)
    max_events = collect_extrema(maxima, [1.0, 0.0]; tspan=(0.0, 20.0), max_events=3)
    @test length(min_events) == 3
    @test length(max_events) == 3
    @test all(event.value < -0.999 for event in min_events)
    @test all(event.value > 0.999 for event in max_events)

    tangent_events = collect_tangent_extrema_rk4(
        minima,
        [1.0, 0.0],
        [0.0, 1.0];
        dt=0.01,
        t_end=20.0,
        max_events=3,
    )
    @test length(tangent_events) == 3
    @test all(event.sign == -1 for event in tangent_events)
end

