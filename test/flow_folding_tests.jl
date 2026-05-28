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

include("../flow_folding/examples/rossler_y_minima_tangent_contours.jl")

@testset "flow_folding scan contours" begin
    temp_dir = mktempdir()
    scan_path = joinpath(temp_dir, "scan.tsv")
    contour_dir = joinpath(temp_dir, "contours")
    open(scan_path, "w") do io
        println(io, "a\tc\tb\tstatus\tevents\tword\tcode\tperiod\tgamma\tmax_time\tfirst_time\tlast_time\tmin_y\tmax_y")
        println(io, "0.30\t2.0\t0.3\tok\t3\t101\t5\t3\t0.625\t80\t1\t3\t-2\t-1")
        println(io, "0.30\t3.0\t0.3\tok\t3\t001\t1\t3\t0.5\t80\t1\t3\t-3\t-1")
        println(io, "0.30\t4.0\t0.3\tok\t3\t011\t3\t3\t0.75\t80\t1\t3\t-4\t-1")
        println(io, "0.40\t2.0\t0.3\tok\t3\t100\t4\t3\t0.125\t80\t1\t3\t-2\t-1")
        println(io, "0.40\t3.0\t0.3\tmax_time\t1\t0\t0\t0\tNaN\t80\t1\t1\t-3\t-3")
        println(io, "0.40\t4.0\t0.3\tok\t3\t111\t7\t1\t0.875\t80\t1\t3\t-4\t-1")
        println(io, "0.50\t2.0\t0.3\tok\t3\t000\t0\t1\t0\t80\t1\t3\t-2\t-1")
        println(io, "0.50\t3.0\t0.3\tok\t3\t010\t2\t3\t0.25\t80\t1\t3\t-3\t-1")
        println(io, "0.50\t4.0\t0.3\tok\t3\t110\t6\t3\t0.375\t80\t1\t3\t-4\t-1")
    end

    write_all_contours(scan_path; output_dir=contour_dir, stem="test_scan")
    @test isfile(joinpath(contour_dir, "test_scan_all_symbol_contours.svg"))
    @test isfile(joinpath(contour_dir, "test_scan_word_boundary_contours.svg"))
    @test isfile(joinpath(contour_dir, "test_scan_prefix03_contours.svg"))
    @test isfile(joinpath(contour_dir, "test_scan_symbol03_contours.svg"))
    summary = read(joinpath(contour_dir, "test_scan_contour_summary.tsv"), String)
    @test occursin("max_time_limited_points\t1", summary)
    @test occursin("word_length\t3", summary)
end
